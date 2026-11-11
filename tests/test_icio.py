"""국제 산업연관 엔진 검증 (OECD ICIO)."""
import io

import numpy as np
import pytest
from openpyxl import load_workbook

from core.excel import to_excel
from core.result import Result
from engines.io import icio as I

AV = I.available()
pytestmark = pytest.mark.skipif(not AV, reason="ICIO 자료 없음")


@pytest.fixture(scope="module")
def t():
    ed = list(AV)[-1]
    return I.load(ed, AV[ed][-1])


def test_identities(t):
    # x = Z1 + Y1 (반올림 오차 범위)
    assert np.abs(t.Z.sum(1) + t.Y.sum(1) - t.x).max() < 5
    # L(Z1+Y1) 재현: L y_total = x
    assert np.allclose(t.Lmul(t.Y.sum(1)), t.x, rtol=1e-6, atol=5)


def test_demand_linear(t):
    s = [{"country": "CHN", "component": "GFCF", "origin": "전체", "industries": [], "pct": -5}]
    dy1 = I.build_demand(t, s)
    dy2 = I.build_demand(t, [dict(s[0], pct=-10)])
    assert np.allclose(2 * dy1, dy2)
    ct, cells, dva, dx, summ = I.demand_effect(t, dy1)
    # 부가가치 + 생산물세 + ... 의 근사: 세계 부가가치 감소 ≤ 최종수요 감소
    assert dva.sum() < 0 and abs(dva.sum()) <= abs(dy1.sum()) * 1.001
    assert ct.loc[ct["국가코드"] == "CHN", "GDP 대비(%)"].iloc[0] < -1.0


def test_tariff_zero_and_sign(t):
    ct, *_ = I.tariff_effect(t, ["USA"], [I.WORLD], [], 0.0, 2.0, 0.3, 0.3)
    assert np.allclose(ct["부가가치 변화(백만 달러)"], 0)
    ct, cells, dva, dx, summ, info = I.tariff_effect(t, ["USA"], ["CHN"], [s for s in t.industries if s[0] == "C"],
                                                     25.0, 2.0, 0.0, 0.0)
    assert info["r"] == pytest.approx(1 - 1.25 ** -2)
    assert ct.loc[ct["국가코드"] == "CHN", "부가가치 변화(백만 달러)"].iloc[0] < 0
    assert ct.loc[ct["국가코드"] == "USA", "소비자물가(%)"].iloc[0] > 0


def test_price_homogeneity(t):
    """모든 셀 가격을 10% 고정하면 그대로 10%."""
    ct, cells, dp, summ = I.price_effect(t, {i: 10.0 for i in range(t.N) if t.x[i] > 0})
    assert np.allclose(dp[t.x > 0], 0.10)


def test_supply_matches_direct(t):
    j = t.idx("TWN", "C26")
    ct, cells, fwd, bwd, summ = I.supply_effect(t, {j: 50.0})
    assert fwd[j] == pytest.approx(-0.5 * t.x[j], rel=1e-6)
    assert bwd[j] <= -0.5 * t.x[j] + 1e-6


def test_gvc_shares(t):
    g = I.gvc_indicators(t)
    kor = g[g["국가코드"] == "KOR"].iloc[0]
    # DVA + FVA 는 총수출의 90~100% (생산물세 몫 제외)
    tot = kor["국내 부가가치(DVA) 비중(%)"] + kor["해외 부가가치(FVA) 비중(%)"]
    assert 90 < tot <= 100.5
    assert 20 < kor["해외 부가가치(FVA) 비중(%)"] < 40


def test_excel_large_table(t):
    dy = I.build_demand(t, [{"country": "USA", "component": "HFCE", "origin": "전체", "industries": [], "pct": -1}])
    ct, cells, *_ = I.demand_effect(t, dy)
    res = Result(module="국제 산업연관표 분석", analysis="테스트", tables={"국가별": ct, "셀": cells})
    wb = load_workbook(io.BytesIO(to_excel(res)), read_only=True)
    assert wb["셀"].max_row == t.N + 1


def test_kww_identity():
    """KWW 분해: 일곱 구성요소 합 = 총수출, 음수 없음, 비중 합 100."""
    from engines.io import icio as I
    av = I.available()
    ed = list(av)[-1]
    k = I.kww(ed, av[ed][-1])
    parts = k[[c + "(백만 달러)" for c in I.KWW_COLS]]
    assert np.allclose(parts.sum(axis=1), k["총수출(백만 달러)"], rtol=1e-8, atol=1e-3)
    assert (parts.to_numpy() > -1e-6).all()
    sh = k[["부가가치 수출(VAX) 비중(%)", "귀환 국내 부가가치(RDV) 비중(%)", "해외 부가가치(FVA) 비중(%)", "순수 이중계산(PDC) 비중(%)"]].sum(axis=1)
    assert np.allclose(sh[k["총수출(백만 달러)"] > 0], 100)
