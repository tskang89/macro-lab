"""공통 차트 (Plotly) — 하나의 디자인 체계.

색: 검증한 범주형 팔레트 8색을 정해진 순서로 쓴다(dataviz 스킬 기준 팔레트; 색각이상 시뮬레이션 인접 쌍 ΔE ≥ 8,
    밝은 화면·어두운 화면 각각 따로 고른 단계). 9번째 색은 만들지 않고, 9개가 넘으면 개별 선은 회색 맥락선으로 그린다.
    양(+)은 파랑, 음(−)은 빨강의 발산형 한 쌍, 가운데는 회색.
표시: 선 2px, 강조선 3px(본문 글자색), 격자는 머리카락 굵기 실선, 막대 끝 4px 둥글게, 쌓은 막대 사이 화면색 틈.
글자는 데이터 색을 쓰지 않고 글자색(주·보조·흐림)만 쓴다. 범례는 2개 계열 이상일 때 위쪽에.
밝은/어두운 테마는 Streamlit 의 현재 테마(st.context.theme.type)를 읽어 고른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FONT = '"Pretendard Variable", Pretendard, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif'

TOKENS = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e9e8e3", "axis": "#c3c2b7", "surface": "#ffffff",
        "band": "rgba(137,135,129,0.16)", "context": "rgba(137,135,129,0.55)",
        "fc_fill": "rgba(137,135,129,0.10)",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "pos": "#3987e5", "neg": "#e66767", "mid": "#383835",
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#45453f", "surface": "#14171c",
        "band": "rgba(195,194,183,0.14)", "context": "rgba(195,194,183,0.45)",
        "fc_fill": "rgba(195,194,183,0.08)",
    },
}
SHADE_TEXT = "전망"


def mode() -> str:
    try:
        import streamlit as st
        t = st.context.theme.type
        return "dark" if t == "dark" else "light"
    except Exception:  # noqa: BLE001  (스크립트 밖·테스트)
        return "light"


def tk() -> dict:
    return TOKENS[mode()]


# 예전 이름 호환 (밝은 테마 값)
POS, NEG, NEUTRAL = TOKENS["light"]["pos"], TOKENS["light"]["neg"], TOKENS["light"]["muted"]
SEQ = PALETTE = TOKENS["light"]["series"]


def _style(fig: go.Figure, title: str = "", height: int = 400, legend: bool = True, unit: str | None = None,
           margin: dict | None = None) -> go.Figure:
    """모든 차트 공통 외형."""
    t = tk()
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15, color=t["ink"], family=FONT, weight=600)) if title else None,
        height=height,
        margin=margin or dict(l=8, r=28, t=72 if (title and legend) else 48 if title else 16, b=8),
        font=dict(family=FONT, size=12, color=t["ink2"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", title=None,
                    font=dict(size=12, color=t["ink2"]), bgcolor="rgba(0,0,0,0)", itemwidth=30,
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(family=FONT, size=12, color=t["ink"])),
        bargap=0.35, barcornerradius=4,
    )
    ax = dict(showgrid=True, gridcolor=t["grid"], gridwidth=1, zeroline=False, showline=False,
              tickfont=dict(size=11, color=t["muted"]), title_font=dict(size=11, color=t["muted"]),
              ticks="", automargin=True)
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    if unit is not None:
        fig.update_yaxes(title_text=unit)
    n_leg = sum(1 for tr in fig.data if tr.showlegend is not False)
    if legend and n_leg > 6:                     # 범례가 여러 줄이면 제목과 겹치지 않게 그림 아래로
        fig.update_layout(legend=dict(y=-0.08, yanchor="top"), margin=dict(t=48 if title else 16))
    return fig


def _baseline(fig, y: float = 0.0):
    fig.add_hline(y=y, line_color=tk()["axis"], line_width=1, layer="below")


# ---------------------------------------------------------------------------
# 단면(부문·나라) 차트
# ---------------------------------------------------------------------------
def bar_top(df: pd.DataFrame, label: str, value: str, n: int = 15, title: str = "", unit: str = ""):
    t = tk()
    d = df[[label, value]].copy()
    d = d.reindex(d[value].abs().sort_values(ascending=False).index[:n]).iloc[::-1]
    colors = [t["pos"] if v >= 0 else t["neg"] for v in d[value]]
    fig = go.Figure(go.Bar(x=d[value], y=d[label], orientation="h", marker_color=colors, marker_line_width=0,
                           hovertemplate="%{y}<br><b>%{x:,.3f}</b>" + unit + "<extra></extra>"))
    _style(fig, title, max(300, 26 * len(d) + 90), legend=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=t["ink2"]))
    fig.update_xaxes(title_text=unit)
    fig.add_vline(x=0, line_color=t["axis"], line_width=1)
    return fig


def bar_stacked(df: pd.DataFrame, label: str, values: list[str], title: str = "", n: int = 15, sort_by: str | None = None):
    t = tk()
    sort_by = sort_by or values[0]
    d = df.reindex(df[sort_by].abs().sort_values(ascending=False).index[:n]).iloc[::-1]
    fig = go.Figure()
    for i, v in enumerate(values):
        fig.add_bar(x=d[v], y=d[label], name=v.strip(), orientation="h", marker_color=t["series"][i % 8],
                    marker_line=dict(color=t["surface"], width=1.5),
                    hovertemplate="%{y}<br>" + v.strip() + " <b>%{x:,.3f}</b><extra></extra>")
    _style(fig, title, max(300, 26 * len(d) + 110))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=t["ink2"]))
    fig.add_vline(x=0, line_color=t["axis"], line_width=1)
    return fig


def heatmap(mat: pd.DataFrame, title: str = "", unit: str = "", zmid: float = 0.0):
    t = tk()
    z = mat.to_numpy(float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
    scale = [[0.0, t["neg"]], [0.5, t["mid"]], [1.0, t["pos"]]]       # 발산형: 빨강 − 회색 − 파랑
    fig = px.imshow(mat, color_continuous_scale=scale, zmin=-lim, zmax=lim, aspect="auto", labels=dict(color=unit))
    _style(fig, title, max(380, 22 * len(mat) + 140), legend=False)
    fig.update_xaxes(side="top", tickangle=-45, showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_traces(xgap=1, ygap=1)
    fig.update_coloraxes(colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=t["muted"], size=11),
                                       title=dict(font=dict(color=t["muted"], size=11))))
    return fig


def scatter_quadrant(df: pd.DataFrame, x: str, y: str, label: str, size: str | None = None, title: str = ""):
    t = tk()
    fig = px.scatter(df, x=x, y=y, text=label, size=size, size_max=36, color_discrete_sequence=[t["series"][0]])
    fig.add_hline(y=1, line_color=t["axis"], line_width=1)
    fig.add_vline(x=1, line_color=t["axis"], line_width=1)
    fig.update_traces(textposition="top center", textfont=dict(size=10, color=t["ink2"]),
                      marker=dict(opacity=0.85, line=dict(color=t["surface"], width=2)))
    _style(fig, title, 560, legend=False)
    return fig


def lines(df: pd.DataFrame, x: str, ys: list[str], title: str = ""):
    t = tk()
    fig = go.Figure()
    for i, c in enumerate(ys):
        fig.add_scatter(x=df[x], y=df[c], name=c, mode="lines+markers", line=dict(color=t["series"][i % 8], width=2),
                        marker=dict(size=8, line=dict(color=t["surface"], width=2)))
    _style(fig, title, 380, legend=len(ys) > 1)
    # x축은 범주형 — 첫 관측 연도에서 시작해 마지막 관측 연도로 끝난다
    fig.update_xaxes(type="category", showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# 시계열 (분기·연간) — x축 눈금은 마지막 관측 시점에서 거꾸로 맞춘다
# ---------------------------------------------------------------------------
def end_ticks(labels: list[str], step: int) -> list[str]:
    """마지막 라벨에서 step 간격으로 거꾸로 고른 눈금 (마지막 관측 시점이 항상 축 끝에 표시됨)."""
    n = len(labels)
    return [labels[i] for i in range(n - 1, -1, -step)][::-1]


def _ts_layout(fig, labels, title, unit, height, step, shade=None, legend=True):
    t = tk()
    step = step or max(1, int(np.ceil(len(labels) / 12)))
    if shade is not None and shade in labels:
        i = labels.index(shade)
        fig.add_vrect(x0=i - 0.5, x1=len(labels) - 0.5, fillcolor=t["fc_fill"], line_width=0, layer="below",
                      annotation_text=SHADE_TEXT, annotation_position="top left",
                      annotation_font=dict(size=11, color=t["muted"], family=FONT))
    _style(fig, title, height, legend=legend, unit=unit)
    fig.update_xaxes(type="category", tickmode="array", tickvals=end_ticks(labels, step), tickangle=0, showgrid=False,
                     range=[-0.5, len(labels) - 0.5 + max(1.0, len(labels) * 0.012)])   # 마지막 눈금 라벨이 잘리지 않게 여백
    fig.update_layout(hovermode="x unified")
    return fig


def ts_lines(df: pd.DataFrame, x: str, ys: list[str], title: str = "", unit: str = "%", height: int = 420,
             step: int | None = None, emphasize: str | None = None, dashed: tuple[str, ...] = (),
             band: tuple[str, str] | None = None, zero: bool = True, shade: str | None = None,
             slots: dict[str, int] | None = None, band_name: str = "방법 간 범위"):
    """slots: 계열 이름 → 범주형 색 번호(0~7). 주면 어떤 조합으로 그려도 같은 대상은 같은 색(색은 순위가 아니라 대상을 따름)."""
    t = tk()
    fig = go.Figure()
    labels = df[x].astype(str).tolist()
    if band:
        lo, hi = band
        fig.add_scatter(x=labels, y=df[hi], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=labels, y=df[lo], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=t["band"], name=band_name, hoverinfo="skip")
    others = [c for c in ys if c != emphasize and c not in dashed]
    crowded = len(others) > 8                    # 9색 이상은 만들지 않음 → 개별 선은 회색 맥락선
    slot = 0
    for c in ys:
        if c == emphasize:
            line = dict(width=3, color=t["ink"])
        elif c in dashed:
            line = dict(width=1.6, color=t["muted"], dash="dot")
        elif crowded:
            line = dict(width=1.2, color=t["context"])
        else:
            k = slots[c] if slots and c in slots else slot
            line = dict(width=2, color=t["series"][k % 8])
            slot += 1
        fig.add_scatter(x=labels, y=df[c], name=c, mode="lines", line=line,
                        hovertemplate="%{y:.2f}" + unit + "<extra>" + c + "</extra>")
    if zero:
        _baseline(fig)
    return _ts_layout(fig, labels, title, unit, height, step, shade, legend=len(ys) + bool(band) > 1)


def ts_bar_line(df: pd.DataFrame, x: str, bar: str, lines_: list[str], title: str = "", unit: str = "%",
                height: int = 380, step: int | None = None, shade: str | None = None):
    t = tk()
    fig = go.Figure()
    labels = df[x].astype(str).tolist()
    colors = [t["pos"] if (v or 0) >= 0 else t["neg"] for v in df[bar].fillna(0)]
    fig.add_bar(x=labels, y=df[bar], name=bar, marker_color=colors, marker_line_width=0,
                hovertemplate="%{y:.2f}" + unit + "<extra>" + bar + "</extra>")
    for i, c in enumerate(lines_):
        fig.add_scatter(x=labels, y=df[c], name=c, mode="lines", line=dict(width=2, color=t["series"][(i + 2) % 8]))
    _baseline(fig)
    fig = _ts_layout(fig, labels, title, unit, height, step, shade, legend=bool(lines_))
    fig.update_layout(bargap=0.15)
    return fig


def ts_stacked(df: pd.DataFrame, x: str, parts: list[str], total: str | None = None, title: str = "",
               unit: str = "%p", height: int = 420, step: int | None = None, shade: str | None = None):
    t = tk()
    fig = go.Figure()
    labels = df[x].astype(str).tolist()
    for i, c in enumerate(parts):
        fig.add_bar(x=labels, y=df[c], name=c, marker_color=t["series"][i % 8],
                    marker_line=dict(color=t["surface"], width=1),
                    hovertemplate="%{y:.2f}" + unit + "<extra>" + c + "</extra>")
    if total:
        fig.add_scatter(x=labels, y=df[total], name=total, mode="lines+markers", line=dict(color=t["ink"], width=2),
                        marker=dict(size=7, color=t["ink"], line=dict(color=t["surface"], width=2)),
                        hovertemplate="%{y:.2f}" + unit + "<extra>" + total + "</extra>")
    fig.update_layout(barmode="relative")
    _baseline(fig)
    fig = _ts_layout(fig, labels, title, unit, height, step, shade)
    if len(labels) > 40:
        fig.update_layout(bargap=0.1, barcornerradius=1)
    return fig
