"""시나리오와 무관한 계산을 미리 해서 data/cache 에 저장한다 (core/diskcache.py).

  1. 산업연관표: 연도·분류(대·중·기본부문)별 유발계수·연관지표(가설추출 포함)
  2. 국제산업연관표: 연도별 GVC 묶음(간이 지표, KWW 분해, 77개국 부가가치 원천·산업별·상대국별 표)
  3. 잠재성장률·중립금리: 화면 기본 설정의 추정 결과와 민감도 분석(가정을 하나씩 바꾼 재추정, 20분 안팎)
     — 실행 중인 로컬 앱(--pages URL)을 기본 설정으로 열어
     화면이 실제로 쓰는 것과 똑같은 키로 저장되게 한다(설정을 이 스크립트에서 다시 만들면 어긋날 수 있음).

자료(ECOS·OECD 원자료, 한국은행 전망)나 엔진 코드가 바뀌면 키가 바뀌므로 이 스크립트를 다시 돌리고 data/cache 를 커밋한다.
실행:
  .venv\\Scripts\\python.exe build\\precompute.py                       # 1·2
  .venv\\Scripts\\python.exe build\\precompute.py --pages http://localhost:8501   # 1·2·3 (앱을 먼저 띄워 둘 것)
  .venv\\Scripts\\python.exe build\\precompute.py --pages http://localhost:8501 --prune
      # 위에 더해 이번 실행에서 안 쓰인 옛 캐시 파일 삭제. 앱을 '새로 띄운 직후'에 돌릴 것
      # (이미 떠 있던 앱은 메모리 캐시로 답해 디스크 파일을 건드리지 않으므로 쓰이는 파일까지 지워짐)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import diskcache  # noqa: E402


def io_tables():
    from engines.io import domestic as D
    for level in D.LEVELS + [D.BASIC]:
        for y in D.available_years(level):
            t0 = time.time()
            D.multipliers_table(level, y)
            print(f"  산업연관표 {level} {y}: {time.time() - t0:.1f}초")


def icio_tables():
    from engines.io import icio as I
    for ed, years in I.available().items():
        for y in years:
            t0 = time.time()
            I.gvc_bundle(ed, y)
            I.load.cache_clear()
            I._vax.cache_clear()
            I.kww.cache_clear()
            print(f"  ICIO {ed}판 {y}: {time.time() - t0:.1f}초")


def pages(base: str):
    """기본 설정·주요 묶음으로 화면을 열어 캐시를 채운다."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1440, "height": 1000})

        def wait_done(limit=600):
            t0 = time.time()
            while time.time() - t0 < limit:
                pg.wait_for_timeout(1500)
                if pg.locator('[data-testid="stStatusWidget"]').count() == 0 and \
                        pg.locator('[data-testid="stSpinner"]').count() == 0:
                    pg.wait_for_timeout(1500)
                    if pg.locator('[data-testid="stSpinner"]').count() == 0:
                        return time.time() - t0
            return None

        def sens(tab):
            # 캐시가 없으면 첫 계산이 길어 탭이 늦게 나타난다(wait_done 이 계산 시작 전에 끝날 수 있음)
            t = pg.get_by_role("tab", name=tab, exact=True)
            t.wait_for(timeout=1800000)
            t.click()
            wait_done()
            btn = pg.get_by_role("button", name="민감도 분석 실행")
            if btn.count():
                btn.first.click()
                print(f"    민감도 분석: {wait_done(3600):.0f}초")

        # 잠재성장률: 기본(한국은행 4개 모형) + 민감도 → 전체 방법 묶음
        pg.goto(f"{base}/potential-growth", timeout=600000)
        print(f"  잠재성장률 기본 설정: {wait_done():.0f}초")
        sens("민감도·불확실성")
        radio = pg.get_by_text("전체 12가지 방법", exact=False)
        if radio.count():
            radio.first.click()
            print(f"  잠재성장률 전체 방법: {wait_done():.0f}초")
        # 중립금리: 기본 설정(기대인플레이션 = 근원물가)
        pg.goto(f"{base}/neutral-rate", timeout=600000)
        print(f"  중립금리 기본 설정: {wait_done():.0f}초")
        sens("민감도")
        b.close()


def prune(since: float):
    """이번 실행에서 읽히거나 쓰이지 않은(= 지금 코드·자료로는 안 쓰이는) 캐시 파일 삭제."""
    n = 0
    for dirpath, _, files in os.walk(diskcache.CACHE):
        for f in files:
            path = os.path.join(dirpath, f)
            if os.path.getmtime(path) < since - 1:
                os.remove(path)
                n += 1
    print(f"  옛 캐시 {n}개 삭제")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", help="실행 중인 앱 주소 (예: http://localhost:8501)")
    ap.add_argument("--prune", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    print("1. 산업연관표 유발계수·연관지표")
    io_tables()
    print("2. 국제산업연관표 GVC")
    icio_tables()
    if a.pages:
        print("3. 잠재성장률·중립금리 기본 설정")
        pages(a.pages.rstrip("/"))
    if a.prune:
        if not a.pages:
            print("  --prune 은 --pages 와 함께 써야 합니다(화면 기본 설정 결과까지 확인해야 안 쓰이는 파일을 가릴 수 있음).")
        else:
            prune(t_start)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(diskcache.CACHE) for f in fs)
    print(f"캐시 크기 {size / 1e6:.1f} MB  ({diskcache.CACHE})")


if __name__ == "__main__":
    main()
