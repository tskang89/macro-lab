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
    assert set(M.METHODS) | set(M.AVG_KEYS) == set(ests)
    assert set(M.GROUP_OF) == set(M.METHODS)


def test_recent_potential_growth_plausible(kr, ests):
    ga, gp = A.annual_tables(kr, ests, "2015Q1")
    row = ga.set_index("연도").loc[2024]
    for k, e in ests.items():
        if k in ("linear", "hamilton", "svar"):     # 추세가 거칠거나 구간 고정인 방법 제외
            continue
        assert 1.0 < row[e.name] < 3.5, (e.name, row[e.name])
    # 코로나 충격 2020년 갭은 대부분 방법에서 마이너스
    avg_names = [e.name for k, e in ests.items() if k.startswith("avg")]
    neg = (gp.set_index("연도").loc[2020].drop(labels=avg_names + ["구분"]) < 0).mean()
    assert neg >= 0.8


def test_mv_signs(ests):
    p = ests["mv"].params
    assert p["필립스곡선 갭 계수 κ"] > 0 and p["오쿤 계수 θ"] > 0
    nairu = ests["mv"].extra["자연실업률"]["자연실업률 추정(%)"]
    assert 2 < nairu.iloc[-1] < 5
    # HP 갭과 같은 방향
    both = pd.concat([ests["mv"].gap, ests["hp"].gap], axis=1).dropna()
    assert np.corrcoef(both.T)[0, 1] > 0.7


@pytest.mark.parametrize("key", ["pf", "pfmv"])
def test_pf_contributions_add_up(ests, key):
    e = ests[key]
    comp = e.extra["잠재성장률 기여도"].dropna()
    dtr = e.trend.diff(4).reindex(comp.index)
    assert np.allclose(comp["잠재성장률(로그 근사)"], dtr, atol=1e-8)
    assert {"노동: 근로시간", "노동: 15세이상인구", "자본", "총요소생산성"} <= set(comp.columns)


def test_pf_uses_capital_services_and_hours(kr, ests):
    p = ests["pf"].params
    assert "자본서비스" in p["자본투입"] and "근로시간" in p["잠재노동 정의"]
    # 2001~05년 → 2016~20년 잠재성장률 하락 (한국은행 5.0% → 2.6% 수준)
    ga, _ = A.annual_tables(kr, {"pf": ests["pf"]}, "2001Q1")
    g = ga.set_index("연도")["생산함수 접근"]
    assert 4.3 < g.loc[2001:2005].mean() < 6.0 and 2.0 < g.loc[2016:2020].mean() < 3.2
    # 근로시간 기여도는 마이너스(근로시간 단축)
    c = ests["pf"].extra["잠재성장률 기여도"]["노동: 근로시간"]
    assert c.loc["2005Q1":"2019Q4"].mean() < 0


def test_pf_variants(kr):
    e1 = M.pf(kr, labour="emprate")
    e2 = M.pf(kr, capital="stock", cu_adjust=False)
    for e in (e1, e2):
        assert e.trend.notna().sum() > 80
    assert "생산자본스톡" in e2.params["자본투입"]


def test_lw_signs(ests):
    p = ests["lw"].params
    assert p["IS: 실질금리 갭 계수 a_r"] < 0 and p["필립스: 갭 계수 b_y"] > 0 and p["오쿤 계수 θ"] > 0
    assert p["IS: 갭 AR(1) a1"] + p["IS: 갭 AR(2) a2"] < 0.98
    t = ests["lw"].extra["중립금리·자연실업률"]
    assert t["중립금리 추정(%)"].notna().all() and 2 < t["자연실업률 추정(%)"].iloc[-1] < 5


def test_kalman_matches_statsmodels_local_level():
    import statsmodels.api as sm
    from engines.potential import statespace as SS
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=120)) + rng.normal(size=120)
    mod = sm.tsa.UnobservedComponents(x, level="llevel")
    res = mod.fit(disp=False)
    s2e, s2l = res.params
    M_ = SS.SSM(np.array([[1.0]]), np.zeros((120, 1)), np.array([[s2e]]), np.array([[1.0]]), np.zeros(1),
                np.array([[s2l]]), np.array([x[0]]), np.array([[1e7]]))
    ll = SS.kfilter(x[:, None], M_, burn=1)[0]
    assert abs(ll - res.llf) < 0.5
    sm_ = SS.ksmooth(x[:, None], M_)[0][:, 0]
    assert np.allclose(sm_, res.level["smoothed"], atol=1e-3)


def test_denton_preserves_annual_means():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("bp", pathlib.Path(__file__).parents[1] / "build" / "build_potential.py")
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    a = pd.Series([100.0, 103.0, 101.0, 108.0], index=[2000, 2001, 2002, 2003])
    q = bp.denton(a, 2000, 2003)
    assert np.allclose(q.groupby(q.index.year).mean().to_numpy(), a.to_numpy())


def test_svar_long_run_restriction(ests):
    assert abs(ests["svar"].params["수요 충격의 GDP 장기 효과(0 이어야 함)"]) < 1e-8


def test_ensemble_is_mean_of_group_means(ests):
    for g in M.GROUPS:
        keys = [k for k in ests if M.GROUP_OF.get(k) == g]
        G = pd.concat([ests[k].gap for k in keys], axis=1).mean(axis=1)
        assert np.allclose(G.dropna(), ests[f"avg_{g}"].gap.dropna())
    G = pd.concat([ests[f"avg_{g}"].gap for g in M.GROUPS], axis=1).mean(axis=1)
    assert np.allclose(G.dropna(), ests["avg"].gap.dropna())


def test_simple_average_mode(kr):
    e, _ = A.run(kr, {"methods": ["hp", "uc", "mv"], "avg_mode": "all"})
    G = pd.concat([e[k].gap for k in ("hp", "uc", "mv")], axis=1).mean(axis=1)
    assert np.allclose(G.dropna(), e["avg"].gap.dropna())


def test_upload_parse_and_univariate_only():
    csv = "quarter,gdp\n" + "\n".join(f"{p.year}Q{p.quarter},{100 * 1.01 ** i:.4f}" for i, p in enumerate(IDX))
    df = D.parse_upload(csv.encode())
    assert df.index.freqstr.startswith("Q") and len(df) == len(IDX)
    e, errs = A.run(df, {"start": "1990Q1"})
    assert {"mv", "pf", "svar", "lw", "pfmv"} <= set(errs)
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


# ---------------------------------------------------------------------------
# 끝점 보정: 한국은행 전망 연장
# ---------------------------------------------------------------------------
from engines.potential import forecast as F  # noqa: E402


def _toy_fc():
    return {"horizon": "2027Q4",
            "gdp": {"q_qoq": {"2026Q3": 0.3, "2026Q4": 0.5}, "q_yoy": {"2026Q3": 2.6, "2026Q4": 3.3},
                    "h_yoy": {"2026H2": 3.0, "2027H1": 2.5, "2027H2": 3.2}, "a_yoy": {"2026": 3.3, "2027": 2.9}},
            "urate": {"h": {"2026H2": 2.9, "2027H1": 2.8, "2027H2": 2.8}},
            "emp_change": {"h": {"2026H2": 18, "2027H1": 23, "2027H2": 17}}}


def _toy_df():
    idx = pd.period_range("2015Q1", "2026Q2", freq="Q")
    n = len(idx)
    return pd.DataFrame({"gdp": 1000 * 1.005 ** np.arange(n), "urate": 3.0, "emp": 27000 + 20 * np.arange(n),
                         "lf": 28000 + 20 * np.arange(n), "pop15": 45000 * 1.002 ** np.arange(n),
                         "ksvc": 100 * 1.007 ** np.arange(n)}, index=idx)


def test_forecast_reproduces_half_year_growth():
    df = _toy_df()
    x, how = F.extend(df, _toy_fc(), "yoy")
    assert str(x.index[-1]) == "2027Q4" and x[F.FC_COL].sum() == 6
    g = x["gdp"]
    hy = lambda y, h: (g[f"{y}Q{2 * h - 1}"] + g[f"{y}Q{2 * h}"]) / (g[f"{y - 1}Q{2 * h - 1}"] + g[f"{y - 1}Q{2 * h}"]) - 1  # noqa: E731
    assert abs(100 * hy(2027, 2) - 3.2) < 1e-6           # 두 분기가 비어 있는 반기: 반기 전망 재현
    assert abs(100 * hy(2027, 1) - 2.5) < 1e-6
    assert abs(100 * (g["2026Q3"] / g["2025Q3"] - 1) - 2.6) < 1e-9   # 분기 경로 전년동기비
    assert np.isclose(x.loc["2027Q1", "emp"], df.loc["2026Q1", "emp"] + 230)
    assert x.loc["2027Q1", "urate"] == 2.8
    assert np.isclose(x.loc["2027Q1", "lf"], x.loc["2027Q1", "emp"] / (1 - 0.028))
    r = np.log(df["ksvc"].iloc[-1] / df["ksvc"].iloc[-13]) / 12
    assert np.isclose(np.log(x.loc["2027Q4", "ksvc"] / x.loc["2027Q3", "ksvc"]), r)   # 추세 연장
    assert set(how.index) == set(x.index[x[F.FC_COL] == 1])


def test_forecast_qoq_mode_and_actuals_untouched():
    df = _toy_df()
    x, _ = F.extend(df, _toy_fc(), "qoq")
    assert np.isclose(x.loc["2026Q3", "gdp"] / x.loc["2026Q2", "gdp"], 1.003)
    pd.testing.assert_frame_equal(x.loc[:"2026Q2", df.columns], df.astype(float), check_freq=False)


def test_forecast_korea_keeps_history_and_marks_tables(kr):
    fc = F.load_bok()
    x, _ = F.extend(kr, fc)
    assert F.last_actual(x) == kr["gdp"].dropna().index[-1]
    ests, errs = A.run(x, {"methods": ["hp", "lw", "pfmv", "pf"]})
    assert not errs
    base, _ = A.run(kr, {"methods": ["lw", "pfmv"]})
    for k in ("lw", "pfmv"):       # 모수는 실적 구간에서만 추정 → 과거 추정치가 크게 바뀌지 않음
        a0 = M.annual_growth(base[k].trend).loc[2005:2022]
        a1 = M.annual_growth(ests[k].trend).loc[2005:2022]
        assert (a0 - a1).abs().max() < 0.3
        assert "평활에만" in ests[k].params["표본"]
    ga, _ = A.annual_tables(x, ests, "2020Q1")
    assert ga.set_index("연도").loc[int(fc["horizon"][:4]), "구분"] == "전망 포함"
    L = A.latest(x, ests)
    assert L["last"] == F.last_actual(x)
    cmp_ = F.compare(x, fc)
    h = cmp_.set_index("기간")
    y2 = fc["horizon"][:4]
    assert abs(h.loc[f"{y2}년 하반기", "연장 경로(%)"] - fc["gdp"]["h_yoy"][f"{y2}H2"]) < 0.01


def test_tail_and_shown():
    df = _toy_df()
    x, _ = F.extend(df, _toy_fc(), "yoy")
    t = F.add_tail(x, 2.0, 12)
    assert (t[F.FC_COL] == F.TAIL).sum() == 12
    g = t["gdp"].pct_change().loc[t[F.FC_COL] == F.TAIL]
    assert np.allclose(g, 1.02 ** 0.25 - 1)
    assert F.shown(t).index[-1] == x.index[-1]
    assert F.first_forecast(t) == x.index[F.FC_COL in x and (x[F.FC_COL] == 1).argmax()]


def test_tfp_cut_extends_trend_linearly(kr):
    e = M.pf(kr, extend=12, tfp_cut="2024Q4")
    tr = e.extra["생산함수 투입"]["추세 TFP"]
    d = tr.loc["2025Q1":].diff().dropna()
    assert np.allclose(d, d.iloc[0])                 # 고정 시점 뒤는 일정한 기울기
    assert "2024Q4" in e.params["TFP 추세"]
    base = M.pf(kr, extend=12)
    assert np.allclose(tr.loc[:"2015Q4"], base.extra["생산함수 투입"]["추세 TFP"].loc[:"2015Q4"], atol=0.05)


def test_expected_inflation_is_lagged_yoy():
    """근원물가 기대 = 전년동기비(t−1) (HLW: 직전 4분기 평균 상승률). 당분기 물가가 들어가면 필립스곡선 식별이 깨짐."""
    df, _ = D.load_korea()
    d = df.drop(columns=["pie"])
    pe = M.expected_inflation(d)
    yoy = (d["core"] / d["core"].shift(4) - 1) * 100
    assert np.allclose(pe.dropna(), yoy.shift(1).reindex(pe.dropna().index))


def test_base_rate_is_quarter_average():
    """기준금리는 분기 평균(ECOS 분기 값은 분기 말 값): 2026Q3 = (2.75+3+3)/3."""
    df, _ = D.load_korea()
    if pd.Period("2026Q3", freq="Q") in df.index:
        assert abs(df.loc[pd.Period("2026Q3", freq="Q"), "rate"] - (2.75 + 3 + 3) / 3) < 1e-4


def test_forecast_extension_no_artificial_dip():
    """기본(전기비) 연장은 분기 경로의 전기비를 실적 수준에 이어 붙이므로 전망 분기 전기비가 경로와 같고 음수 되돌림이 없음."""
    df, _ = D.load_korea()
    fc = F.load_bok()
    dfx, _ = F.extend(df, fc)
    g = dfx["gdp"]
    for q, v in fc["gdp"]["q_qoq"].items():
        p = pd.Period(q, freq="Q")
        if p > F.last_actual(dfx):
            assert abs((g[p] / g[p - 1] - 1) * 100 - v) < 1e-6


def test_half_year_fill_matches_target_when_one_quarter_known():
    """반기 한 분기가 이미 정해졌을 때 나머지 분기로 반기 전년동기비를 정확히 맞춤."""
    idx = pd.period_range("2024Q1", "2025Q3", freq="Q")
    s = pd.Series(np.linspace(100, 106, len(idx)), index=idx)
    fc = {"q_yoy": {}, "q_qoq": {}, "h_yoy": {"2025H2": 3.0}, "a_yoy": {}}
    out, _ = F._extend_gdp(s, fc, [pd.Period("2025Q4", freq="Q")], mode="yoy")
    h = out[pd.Period("2025Q3", freq="Q")] + out[pd.Period("2025Q4", freq="Q")]
    h0 = out[pd.Period("2024Q3", freq="Q")] + out[pd.Period("2024Q4", freq="Q")]
    assert abs((h / h0 - 1) * 100 - 3.0) < 1e-6
