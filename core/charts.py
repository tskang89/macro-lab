"""공통 차트 (Plotly). 색은 한 가지 체계로: 음(-)은 붉은 계열, 양(+)은 푸른 계열."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

POS = "#2E6BB0"
NEG = "#C8553D"
NEUTRAL = "#7F8C9A"
SEQ = ["#1F3864", "#2E6BB0", "#5B9BD5", "#9DC3E6", "#C8553D", "#E59866", "#7F8C9A", "#A9CCE3"]


def bar_top(df: pd.DataFrame, label: str, value: str, n: int = 15, title: str = "", unit: str = ""):
    d = df[[label, value]].copy()
    d = d.reindex(d[value].abs().sort_values(ascending=False).index[:n]).iloc[::-1]
    colors = [POS if v >= 0 else NEG for v in d[value]]
    fig = go.Figure(go.Bar(x=d[value], y=d[label], orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>%{x:,.3f}" + unit + "<extra></extra>"))
    fig.update_layout(title=title, height=max(300, 26 * len(d) + 90), margin=dict(l=10, r=20, t=50, b=30),
                      xaxis_title=unit, yaxis_title=None)
    return fig


def bar_stacked(df: pd.DataFrame, label: str, values: list[str], title: str = "", n: int = 15, sort_by: str | None = None):
    sort_by = sort_by or values[0]
    d = df.reindex(df[sort_by].abs().sort_values(ascending=False).index[:n]).iloc[::-1]
    fig = go.Figure()
    for i, v in enumerate(values):
        fig.add_bar(x=d[v], y=d[label], name=v.strip(), orientation="h", marker_color=SEQ[i % len(SEQ)])
    fig.update_layout(barmode="relative", title=title, height=max(300, 26 * len(d) + 110),
                      margin=dict(l=10, r=20, t=50, b=30), legend=dict(orientation="h", y=-0.12))
    return fig


def heatmap(mat: pd.DataFrame, title: str = "", unit: str = "", zmid: float = 0.0):
    z = mat.to_numpy(float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
    fig = px.imshow(mat, color_continuous_scale="RdBu", zmin=-lim, zmax=lim, aspect="auto",
                    labels=dict(color=unit))
    fig.update_layout(title=title, height=max(380, 22 * len(mat) + 140), margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(side="top", tickangle=-45)
    return fig


def scatter_quadrant(df: pd.DataFrame, x: str, y: str, label: str, size: str | None = None, title: str = ""):
    fig = px.scatter(df, x=x, y=y, text=label, size=size, size_max=40, color_discrete_sequence=[POS])
    fig.add_hline(y=1, line_dash="dot", line_color=NEUTRAL)
    fig.add_vline(x=1, line_dash="dot", line_color=NEUTRAL)
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(title=title, height=560, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def lines(df: pd.DataFrame, x: str, ys: list[str], title: str = ""):
    fig = go.Figure()
    for i, c in enumerate(ys):
        fig.add_scatter(x=df[x], y=df[c], name=c, mode="lines+markers", line_color=SEQ[i % len(SEQ)])
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    # x축은 범주형 — 첫 관측 연도에서 시작해 마지막 관측 연도로 끝난다
    fig.update_xaxes(type="category")
    return fig


# ---------------------------------------------------------------------------
# 시계열 (분기·연간) — x축 눈금은 마지막 관측 시점에서 거꾸로 맞춘다
# ---------------------------------------------------------------------------
PALETTE = ["#1F3864", "#2E6BB0", "#C8553D", "#E59866", "#5B9BD5", "#7F8C9A", "#8E44AD", "#27AE60",
           "#D4AC0D", "#16A085", "#A04000", "#2C3E50"]


def end_ticks(labels: list[str], step: int) -> list[str]:
    """마지막 라벨에서 step 간격으로 거꾸로 고른 눈금 (마지막 관측 시점이 항상 축 끝에 표시됨)."""
    n = len(labels)
    return [labels[i] for i in range(n - 1, -1, -step)][::-1]


def _ts_layout(fig, labels, title, unit, height, step):
    step = step or max(1, int(np.ceil(len(labels) / 12)))
    fig.update_xaxes(type="category", tickmode="array", tickvals=end_ticks(labels, step), tickangle=0)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=unit,
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    return fig


def ts_lines(df: pd.DataFrame, x: str, ys: list[str], title: str = "", unit: str = "%", height: int = 420,
             step: int | None = None, emphasize: str | None = None, dashed: tuple[str, ...] = (),
             band: tuple[str, str] | None = None, zero: bool = True):
    fig = go.Figure()
    labels = df[x].astype(str).tolist()
    if band:
        lo, hi = band
        fig.add_scatter(x=labels, y=df[hi], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=labels, y=df[lo], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(127,140,154,0.18)", name="방법 간 범위", hoverinfo="skip")
    for i, c in enumerate(ys):
        width = 3 if c == emphasize else 1.6
        color = "#111111" if c == emphasize else PALETTE[i % len(PALETTE)]
        fig.add_scatter(x=labels, y=df[c], name=c, mode="lines",
                        line=dict(width=width, color=color, dash="dot" if c in dashed else "solid"),
                        hovertemplate="%{y:.2f}" + unit + "<extra>" + c + "</extra>")
    if zero:
        fig.add_hline(y=0, line_color="#999999", line_width=1)
    return _ts_layout(fig, labels, title, unit, height, step)


def ts_bar_line(df: pd.DataFrame, x: str, bar: str, lines_: list[str], title: str = "", unit: str = "%",
                height: int = 380, step: int | None = None):
    fig = go.Figure()
    labels = df[x].astype(str).tolist()
    colors = [POS if (v or 0) >= 0 else NEG for v in df[bar].fillna(0)]
    fig.add_bar(x=labels, y=df[bar], name=bar, marker_color=colors, opacity=0.85,
                hovertemplate="%{y:.2f}" + unit + "<extra>" + bar + "</extra>")
    for i, c in enumerate(lines_):
        fig.add_scatter(x=labels, y=df[c], name=c, mode="lines", line=dict(width=2, color=SEQ[i % len(SEQ)]))
    fig.add_hline(y=0, line_color="#999999", line_width=1)
    return _ts_layout(fig, labels, title, unit, height, step)


def ts_stacked(df: pd.DataFrame, x: str, parts: list[str], total: str | None = None, title: str = "",
               unit: str = "%p", height: int = 420, step: int | None = None):
    fig = go.Figure()
    labels = df[x].astype(str).tolist()
    for i, c in enumerate(parts):
        fig.add_bar(x=labels, y=df[c], name=c, marker_color=SEQ[i % len(SEQ)])
    if total:
        fig.add_scatter(x=labels, y=df[total], name=total, mode="lines+markers", line=dict(color="#111111", width=2))
    fig.update_layout(barmode="relative")
    fig.add_hline(y=0, line_color="#999999", line_width=1)
    return _ts_layout(fig, labels, title, unit, height, step)
