"""추정 불확실성(칼만 평활 + 모수 오차)과 민감도 변형 목록."""
import numpy as np

from engines.potential import analysis as A
from engines.potential import sensitivity as S
from engines.potential import statespace as SS


def _local_level(n=120, seed=1):
    rng = np.random.default_rng(seed)
    mu = np.cumsum(rng.normal(0, 0.3, n))
    y = (mu + rng.normal(0, 1.0, n))[:, None]

    def build(x):
        sl, se = np.exp(x)
        return SS.SSM(np.array([[1.0]]), np.zeros((n, 1)), np.array([[se ** 2]]), np.array([[1.0]]), np.zeros(1),
                      np.array([[sl ** 2]]), np.array([y[0, 0]]), np.array([[10.0]]))
    return y, build


def test_uncertainty_total_exceeds_filter():
    y, build = _local_level()
    bounds = [(-6, 2), (-6, 2)]
    xh, _, _ = SS.fit(y, build, np.log([0.5, 0.5]), bounds)
    u = SS.uncertainty(y, y, build, xh, bounds, np.array([[1.0]]), n_draws=100)
    assert u["total"] is not None and u["draws"] >= 30
    assert np.all(np.isfinite(u["total"])) and np.all(u["total"] >= u["filter"] - 1e-12)
    # 평활 오차는 표본 끝에서 가장 큼(한쪽 자료만 있음)
    assert u["filter"][-1, 0] > np.median(u["filter"][:, 0])


def test_flat_parameter_is_fixed():
    """우도와 무관한 모수가 있어도 헤시안을 그 모수를 뺀 채 계산한다."""
    y, build0 = _local_level()

    def build(x):
        return build0(x[:2])               # x[2] 는 우도에 안 들어감
    bounds = [(-6, 2), (-6, 2), (0, 1)]
    xh, _, _ = SS.fit(y, build, np.array([np.log(0.5), np.log(0.5), 0.5]), bounds)
    u = SS.uncertainty(y, y, build, xh, bounds, np.array([[1.0]]), n_draws=60)
    assert u["total"] is not None and 2 not in u["free"]


def test_variant_lists():
    V, base = S.potential_variants(dict(A.DEFAULT_CFG, methods=["hp", "lw", "pfmv", "pf"]), True,
                                   {"mode": "qoq", "tail": 2.0})
    groups = {v.group for v in V}
    assert "자연실업률 변동 크기(분기 %p)" in groups and "끝점 보정" in groups and "노동소득분배율" in groups
    assert all(base[v.group] != v.label for v in V)                 # 기준값과 같은 변형은 없음
    ls = [v for v in V if v.group == "노동소득분배율"]
    assert {v.over["pf_labshare"] for v in ls} == {0.55, 0.67}
    nc = {"pie": "core", "fc": True, "tp": 0.8, "uc_smax": 0.2, "uc_rho": 0.9, "ustar_sd": 0.04, "lw_lam_z": 0.03,
          "lam": 1600.0, "var_window": 40, "g_base": ["2001Q1", "2019Q4"]}
    V2, _ = S.neutral_variants(nc, True)
    assert any(v.over.get("pie") == "survey" for v in V2)
    assert {v.over["tp"] for v in V2 if "tp" in v.over} == {0.4, 1.2}
