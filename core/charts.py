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
