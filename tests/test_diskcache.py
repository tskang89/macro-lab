import pandas as pd

from core import diskcache


def test_disk_cache_roundtrip_and_keys(tmp_path, monkeypatch):
    """같은 인자는 저장된 결과를 다시 쓰고, 인자(데이터프레임 내용)가 바뀌면 다시 계산한다."""
    monkeypatch.setattr(diskcache, "CACHE", str(tmp_path))
    calls = []

    @diskcache.disk_cache("t", code=("core",))
    def f(df, k):
        calls.append(k)
        return df["a"].sum() * k

    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert f(df, 2) == 6 and f(df, 2) == 6
    assert calls == [2]                                 # 두 번째는 디스크에서
    assert f(pd.DataFrame({"a": [1.0, 3.0]}), 2) == 8   # 내용이 다르면 새로 계산
    assert len(calls) == 2
    assert len(list((tmp_path / "t").glob("*.pkl"))) == 2
    assert f.is_cached(df, 2) and not f.is_cached(df, 3)


def test_disk_cache_survives_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(diskcache, "CACHE", str(tmp_path))

    @diskcache.disk_cache("c", code=("core",))
    def g(x):
        return x + 1

    assert g(1) == 2
    for p in (tmp_path / "c").glob("*.pkl"):
        p.write_bytes(b"broken")
    assert g(1) == 2                                    # 손상 파일은 다시 계산
