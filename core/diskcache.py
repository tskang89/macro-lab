"""계산 결과를 디스크(data/cache)에 저장해 두고 다시 꺼내 쓰는 캐시.

Streamlit 의 st.cache_data 는 프로세스 메모리에만 남아 서버가 재시작되면(Streamlit Cloud 는 잠들었다 깨면 재시작) 다시 계산한다.
여기 저장한 결과는 저장소에 함께 올려 두므로, 재시작 직후 첫 화면도 계산 없이 바로 뜬다.

키 = 함수 이름 + 인자(데이터프레임은 내용 해시) + 엔진 소스 코드 해시 + 원자료 파일 해시.
  → 자료나 계산 코드가 바뀌면 키가 달라져 자동으로 다시 계산하고, 옛 파일은 쓰이지 않는다.
읽기 실패(손상·버전 차이)는 조용히 다시 계산한다. 쓰기 실패(읽기 전용 디스크 등)는 무시한다.
미리 채우기: python build/precompute.py
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
ENABLED = os.environ.get("MACROLAB_DISK_CACHE", "1") != "0"


@functools.lru_cache(maxsize=64)
def _file_hash(path: str, mtime: float, size: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash(path: str) -> str:
    st_ = os.stat(path)
    return _file_hash(path, st_.st_mtime, st_.st_size)


@functools.lru_cache(maxsize=16)
def code_hash(*packages: str) -> str:
    """소스(.py) 내용 해시. packages 는 ROOT 기준 폴더(예: 'engines/io') 또는 .py 파일."""
    h = hashlib.sha1()
    for p in packages:
        files = [os.path.join(ROOT, p)] if p.endswith(".py") else             sorted(glob.glob(os.path.join(ROOT, p, "**", "*.py"), recursive=True))
        for f in files:
            h.update(os.path.relpath(f, ROOT).replace("\\", "/").encode())
            with open(f, "rb") as fh:
                h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _arg_token(a) -> str:
    if isinstance(a, pd.DataFrame):
        body = pd.util.hash_pandas_object(a, index=True).to_numpy()
        return "df:" + hashlib.sha1(body.tobytes() + repr(list(a.columns)).encode()).hexdigest()
    if isinstance(a, pd.Series):
        return "s:" + hashlib.sha1(pd.util.hash_pandas_object(a, index=True).to_numpy().tobytes()).hexdigest()
    if isinstance(a, np.ndarray):
        return "a:" + hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
    return json.dumps(a, sort_keys=True, default=str, ensure_ascii=False)


def disk_cache(name: str, code: tuple[str, ...], data_files=None):
    """함수 결과를 data/cache/{name}/{키}.pkl 에 저장·재사용하는 데코레이터.

    code       : 결과에 영향을 주는 소스 폴더들 (해시가 키에 들어감)
    data_files : 인자 → 원자료 파일 경로 목록을 돌려주는 함수(선택). 파일 내용 해시가 키에 들어감.
    """
    def deco(fn):
        @functools.wraps(fn)
        def _path(*args, **kwargs):
            parts = [name, code_hash(*code)] + [_arg_token(a) for a in args] + \
                    [f"{k}={_arg_token(v)}" for k, v in sorted(kwargs.items())]
            if data_files is not None:
                parts += [file_hash(p) for p in data_files(*args, **kwargs)]
            key = hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]
            return os.path.join(CACHE, name, key + ".pkl")

        def wrapper(*args, **kwargs):
            if not ENABLED:
                return fn(*args, **kwargs)
            path = _path(*args, **kwargs)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        out = pickle.load(f)
                    try:
                        os.utime(path)          # 쓰인 파일 표시 (build/precompute.py --prune 이 안 쓰인 옛 파일을 지움)
                    except OSError:
                        pass
                    return out
                except Exception:  # noqa: BLE001 — 손상·버전 차이 → 다시 계산
                    pass
            out = fn(*args, **kwargs)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:  # noqa: BLE001 — 읽기 전용 디스크 등
                pass
            return out
        wrapper.uncached = fn
        wrapper.is_cached = lambda *a, **k: ENABLED and os.path.exists(_path(*a, **k))
        return wrapper
    return deco
