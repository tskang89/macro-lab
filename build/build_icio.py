"""OECD ICIO 원자료(csv in zip) → 표준화 자료(data/processed/icio/icio_{판}_{연도}.npz).

지원: 2023년판(77개 경제권×45개 산업, ~2020), 2025년판(81×50, ~2022) SML 묶음.
  data/raw/icio/ 아래 zip 을 넣고 실행하면 연도별로 변환한다.
  2025년판은 OECD 새 서버가 자동 다운로드를 막아 브라우저로 직접 받아 넣어야 한다.
    https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html
    → 2025 release (regular ICIO) → 2016-2022 등 zip 을 data/raw/icio/ICIO2025_<기간>_SML.zip 으로 저장

산출 (float32 로 저장해 용량 절감)
  countries, industries : 코드
  Z  : 중간거래 (N×N), N = 국가수×산업수, 행·열 순서 = 국가별 산업 블록
  Y  : 최종수요 (N × 국가수×6), 열 순서 = 국가별 FD_CODES
  tls, va, x : 생산물세(순), 부가가치, 총산출 (N)
실행: python build/build_icio.py
"""
from __future__ import annotations

import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "icio")
OUT = os.path.join(ROOT, "data", "processed", "icio")
FD_CODES = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]


def convert(df: pd.DataFrame, edition: str, year: str):
    df = df.set_index(df.columns[0])
    cols = list(df.columns)
    inter = [c for c in cols if "_" in c and c.split("_", 1)[1] not in FD_CODES]
    countries = list(dict.fromkeys(c.split("_", 1)[0] for c in inter))
    industries = list(dict.fromkeys(c.split("_", 1)[1] for c in inter))
    # 일부 판은 중국·멕시코를 CN1/CN2, MX1/MX2 로 나눔 → 합침
    assert len(inter) == len(countries) * len(industries), (len(inter), len(countries), len(industries))
    rows = inter
    Z = df.loc[rows, inter].to_numpy(np.float64)
    fd_cols = [f"{c}_{f}" for c in countries for f in FD_CODES]
    Y = df.loc[rows, fd_cols].to_numpy(np.float64)
    tls = df.loc["TLS", inter].to_numpy(np.float64)
    va = df.loc["VA", inter].to_numpy(np.float64)
    x = df.loc[rows, "OUT"].to_numpy(np.float64)
    xc = df.loc["OUT", inter].to_numpy(np.float64)
    err_r = np.abs(Z.sum(1) + Y.sum(1) - x).max()
    err_c = np.abs(Z.sum(0) + tls + va - xc).max()
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"icio_{edition}_{year}.npz")
    np.savez_compressed(fn, edition=edition, year=year, countries=np.array(countries),
                        industries=np.array(industries), fd_codes=np.array(FD_CODES),
                        Z=Z.astype(np.float32), Y=Y.astype(np.float32),
                        tls=tls, va=va, x=x)
    return fn, len(countries), len(industries), err_r, err_c


def main():
    zips = sorted(glob.glob(os.path.join(RAW, "*.zip")))
    for zp in zips:
        m = re.search(r"ICIO(\d{4})", os.path.basename(zp))
        edition = m.group(1) if m else "2025"
        if not zipfile.is_zipfile(zp):
            print("not a zip, skip", zp)
            continue
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                ym = re.match(r"(\d{4})_SML\.csv$", os.path.basename(name))
                if not ym:
                    continue
                year = ym.group(1)
                fn = os.path.join(OUT, f"icio_{edition}_{year}.npz")
                if os.path.exists(fn):
                    print("skip", fn)
                    continue
                with zf.open(name) as f:
                    df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)
                out = convert(df, edition, year)
                print("built", out[0], f"{out[1]}x{out[2]}", f"rowerr={out[3]:.3g} colerr={out[4]:.3g}",
                      f"{os.path.getsize(out[0]) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
