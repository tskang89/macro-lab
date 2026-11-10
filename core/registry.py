"""modules/ 아래 폴더를 읽어 왼쪽 메뉴를 자동 구성한다.

각 모듈 폴더의 module.py 는 다음 두 가지만 갖추면 된다.
    META = {"title": "...", "group": "...", "order": 10, "icon": ":material/...:",
            "status": "ready" | "planned", "slug": "url-path", "summary": "한 줄 설명"}
    def render() -> None   # 화면 그리기
"""
from __future__ import annotations

import importlib
import os

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIR = os.path.join(ROOT, "modules")

GROUP_ORDER = ["", "산업연관 분석", "성장 분석", "구조 모형", "도움말"]
PAGES: dict = {}          # slug → st.Page (처음 화면의 바로가기용)


def discover() -> list[dict]:
    mods = []
    for name in sorted(os.listdir(MODULE_DIR)):
        path = os.path.join(MODULE_DIR, name, "module.py")
        if not os.path.exists(path):
            continue
        try:
            m = importlib.import_module(f"modules.{name}.module")
        except KeyError:  # 개발 중 자동 재실행이 겹칠 때의 일시적 import 경합
            importlib.invalidate_caches()
            m = importlib.import_module(f"modules.{name}.module")
        meta = dict(m.META)
        meta["render"] = m.render
        meta.setdefault("slug", name.replace("_", "-"))
        mods.append(meta)
    mods.sort(key=lambda d: (GROUP_ORDER.index(d["group"]) if d["group"] in GROUP_ORDER else 99, d.get("order", 50)))
    return mods


def navigation(mods: list[dict]):
    pages: dict[str, list] = {}
    for meta in mods:
        title = meta["title"] + (" (준비 중)" if meta.get("status") == "planned" else "")
        page = st.Page(meta["render"], title=title, icon=meta.get("icon"), url_path=meta["slug"],
                       default=meta.get("default", False))
        PAGES[meta["slug"]] = page
        pages.setdefault(meta["group"], []).append(page)
    return st.navigation(pages)
