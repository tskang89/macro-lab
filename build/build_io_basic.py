"""한국은행 기본부문(380) 산업연관표(실측표) 엑셀 → data/processed/io_kr/io_kr_기본부문_{연도}.npz

원자료 (ECOS 통계검색 > 2.2 산업연관표 > 2.2.1 2020년 실측표 기준 > 2.2.1.1 파일 다운로드, 사용자가 받아 둠)
  data/raw/io_basic/io{연도}_basic.xlsx   : 「2020 실측표」 생산자가격 > 기본부문
                                            (A표_총거래표·국산거래표, 생산유발계수 등 시트)
  data/raw/io_basic/product_class.xlsx    : 「2020 부문분류표」 > 상품분류표 (기본 380 → 소 165 → 중 83 → 대 33)
  data/raw/io_basic/employment.xlsx       : 「고용표」 > 통합소분류 > 취업자수 (상품 기준, 전업환산, 2020~2024년)

ECOS Open API 에는 기본부문 표가 없고(대·중분류만), 파일 다운로드로만 제공된다. 실측표 연도(2020)에만 있다.

산출 (기존 대·중분류 npz 와 같은 키 → engines.io.domestic.KrIO 로 그대로 읽힘)
  Z, F            : 총거래표 중간거래(380×380)·최종수요(380×7), 백만원
  Fd              : 국산거래표 최종수요(380×7)  ← 비경쟁수입형 국산 최종수요(실측)
  imports         : 수입 + 생산물세(수입)
  x, V            : 총산출, 부가가치 4항목(4×380)
  L_bok           : (I − Ad)^-1, Ad = 국산거래표 중간거래 / 총산출 (공표 생산유발계수와 대조 검증)
  Le_bok, Lw_bok  : diag(e)·L, diag(w)·L — 취업·피용자 계수는 소분류(165) 계수를 그 안의 기본부문에 똑같이 적용
  Lv_bok, Lm_bok  : 부가가치·수입 유발계수(열합)
  small/mid/large : 기본부문별 소·중·대분류 코드, 이름
실행: python build/build_io_basic.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "io_basic")
OUT = os.path.join(ROOT, "data", "processed", "io_kr")

FD_CODES = ["9111", "9112", "9121", "9122", "9131", "9132", "9140"]
VA_CODES = ["9610", "9621", "9622", "9630"]
YEARS = ["2020"]


def norm(c) -> str | None:
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return None
    if isinstance(c, (int, float)):
        return f"{int(c):04d}"
    s = str(c).strip()
    return s or None


def read_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """ECOS 엑셀 시트 → 행코드×열코드 표(숫자), 코드→이름."""
    cols = [norm(c) for c in df.iloc[4, 2:]]
    body = df.iloc[6:]
    rows = [norm(c) for c in body.iloc[:, 0]]
    names = {r: str(n).strip() for r, n in zip(rows, body.iloc[:, 1]) if r}
    vals = body.iloc[:, 2:2 + len(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    keep_r = [i for i, r in enumerate(rows) if r]
    keep_c = [j for j, c in enumerate(cols) if c]
    out = pd.DataFrame(vals[np.ix_(keep_r, keep_c)], index=[rows[i] for i in keep_r], columns=[cols[j] for j in keep_c])
    out = out[~out.index.duplicated()]
    out = out.loc[:, ~out.columns.duplicated()]
    return out, names


def classification() -> pd.DataFrame:
    d = pd.read_excel(os.path.join(RAW, "product_class.xlsx"), header=None, dtype=str).iloc[3:]
    d = d[d[0].str.fullmatch(r"\d{4}", na=False) & ~d[0].str.startswith("9", na=False)].copy()
    d[[2, 3, 4, 5, 6, 7]] = d[[2, 3, 4, 5, 6, 7]].ffill()
    d.columns = ["basic", "basic_nm", "small", "small_nm", "mid", "mid_nm", "large", "large_nm"]
    return d.set_index("basic")


def employment(year: str) -> pd.DataFrame:
    """소분류(상품)별 취업자수·피용자수 (명, 전업환산)."""
    d = pd.read_excel(os.path.join(RAW, "employment.xlsx"), sheet_name=0, header=None)
    yrow = d.iloc[4].ffill()
    cols = [j for j in range(2, d.shape[1]) if str(yrow[j]).startswith(year)]
    body = d.iloc[6:]
    code = body.iloc[:, 0].map(lambda c: f"{int(c):03d}" if isinstance(c, (int, float)) and c == c else str(c).strip())
    out = pd.DataFrame({"emp": pd.to_numeric(body.iloc[:, cols[0]], errors="coerce").to_numpy(),
                        "wage": pd.to_numeric(body.iloc[:, cols[1]], errors="coerce").to_numpy()}, index=code.to_numpy())
    return out[out.index.str.fullmatch(r"\d{3}")]


def build(year: str) -> dict:
    path = os.path.join(RAW, f"io{year}_basic.xlsx")
    sh = pd.read_excel(path, sheet_name=["A표_총거래표(생산자)", "A표_국산거래표(생산자)", "생산유발계수"], header=None)
    tot, names = read_table(sh["A표_총거래표(생산자)"])
    dom, _ = read_table(sh["A표_국산거래표(생산자)"])
    pub, _ = read_table(sh["생산유발계수"])
    cl = classification()
    codes = [c for c in cl.index]
    assert len(codes) == 380, len(codes)
    Z = tot.loc[codes, codes].to_numpy()
    F = tot.loc[codes, FD_CODES].to_numpy()
    Zd = dom.loc[codes, codes].to_numpy()
    Fd = dom.loc[codes, FD_CODES].to_numpy()
    x = tot.loc[codes, "9310"].to_numpy()
    x_col = tot.loc["9790", codes].to_numpy()
    imports = (tot.loc[codes, "9321"] + tot.loc[codes, "9322"]).to_numpy()
    V = tot.loc[VA_CODES, codes].to_numpy()
    # 검증
    assert np.allclose(x, x_col, rtol=1e-6, atol=1.0), np.abs(x - x_col).max()
    assert np.allclose(Z.sum(0) + V.sum(0), x, rtol=1e-6, atol=2.0)
    assert np.allclose(Zd.sum(1) + Fd.sum(1), x, rtol=1e-6, atol=2.0), np.abs(Zd.sum(1) + Fd.sum(1) - x).max()
    xs = np.where(x > 0, x, 1.0)
    Ad = Zd / xs[None, :]
    L = np.linalg.inv(np.eye(len(codes)) - Ad)
    Lp = pub.loc[codes, codes].to_numpy()
    err = float(np.abs(L - Lp).max())
    print(f"{year}: 380부문, 총산출 {x.sum() / 1e6:,.1f}조원, 생산유발계수 공표치와 최대 차이 {err:.2e}")
    assert err < 1e-4
    A = Z / xs[None, :]
    Am = A - Ad
    v = V.sum(0) / xs
    # 취업·피용자 계수: 소분류 계수(명/십억원) = 소분류 취업자 / 소분류 총산출
    em = employment(year)
    small = cl["small"].to_numpy()
    xs_small = pd.Series(x, index=small).groupby(level=0).sum()
    miss = sorted(set(small) - set(em.index))
    assert not miss, miss
    e_small = em["emp"] / (xs_small / 1e3)
    w_small = em["wage"] / (xs_small / 1e3)
    e = np.nan_to_num(e_small.reindex(small).to_numpy())
    w = np.nan_to_num(w_small.reindex(small).to_numpy())
    print(f"  취업자 {np.sum(e * x / 1e3) / 1e4:,.0f}만 명(고용표 소분류 합 {em['emp'].sum() / 1e4:,.0f}만 명)")
    return dict(
        year=year, level="기본부문", basis="2020",
        codes=np.array(codes), names=np.array([names.get(c, cl.loc[c, "basic_nm"]) for c in codes]),
        Z=Z, F=F, Fd=Fd, imports=imports, x=x, V=V,
        L_bok=L, Le_bok=np.diag(e) @ L, Lw_bok=np.diag(w) @ L,
        Lv_bok=v @ L, Lm_bok=Am.sum(0) @ L,
        small=cl["small"].to_numpy(str), small_nm=cl["small_nm"].to_numpy(str),
        mid=cl["mid"].to_numpy(str), mid_nm=cl["mid_nm"].to_numpy(str),
        large=cl["large"].to_numpy(str), large_nm=cl["large_nm"].to_numpy(str),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    for y in YEARS:
        if not os.path.exists(os.path.join(RAW, f"io{y}_basic.xlsx")):
            print(f"{y}: 원자료 없음 — 건너뜀")
            continue
        d = build(y)
        np.savez_compressed(os.path.join(OUT, f"io_kr_기본부문_{y}.npz"), **d)


if __name__ == "__main__":
    main()
