"""Plotly visualisations for the CKA connectome.

Three figure factories that all return ``plotly.graph_objects.Figure`` objects:

- :func:`connectome_heatmap` — single ``L x L`` CKA matrix with the cividis
  palette, colour-bar locked to ``[0, 1]`` so panels are comparable.
- :func:`panel` — side-by-side heatmaps for the multi-task contrast, sharing
  axes and colour-bar.
- :func:`delta_heatmap` — the RYS difference plot, diverging palette centred
  at 0 with the design system's deep-orange accent.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACCENT = "#c2410c"


def connectome_heatmap(
    M: pd.DataFrame,
    *,
    title: str = "",
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> go.Figure:
    """Render a single CKA connectome as a heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=M.to_numpy(),
            x=list(M.columns),
            y=list(M.index),
            colorscale="cividis",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "linear CKA"},
            hovertemplate="layer i = %{y}<br>layer j = %{x}<br>CKA = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title="layer i")
    fig.update_xaxes(title="layer j")
    fig.update_layout(
        title=title,
        width=560,
        height=520,
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )
    return fig


def panel(
    matrices: dict[str, pd.DataFrame],
    *,
    title: str = "",
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> go.Figure:
    """Place CKA matrices side-by-side for cross-task comparison."""
    n = len(matrices)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=list(matrices.keys()),
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )
    for i, (name, M) in enumerate(matrices.items(), start=1):
        fig.add_trace(
            go.Heatmap(
                z=M.to_numpy(),
                x=list(M.columns),
                y=list(M.index),
                colorscale="cividis",
                zmin=zmin,
                zmax=zmax,
                showscale=(i == n),
                colorbar={"title": "CKA"} if i == n else None,
                hovertemplate=(
                    f"task = {name}<br>layer i = %{{y}}<br>layer j = %{{x}}"
                    "<br>CKA = %{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="layer j", row=1, col=i)
    fig.update_yaxes(autorange="reversed", title="layer i", row=1, col=1)
    fig.update_layout(
        title=title,
        width=320 * n + 80,
        height=520,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
    )
    return fig


def delta_heatmap(
    M_rys: pd.DataFrame,
    M_base: pd.DataFrame,
    *,
    title: str = "",
    window: tuple[int, int] | None = None,
) -> go.Figure:
    """Plot ``M_rys - M_base`` as a diverging heatmap.

    A solid box is drawn around the duplicated window when one is provided,
    so the predicted off-plateau amplification is visually localisable.
    """
    if not M_rys.index.equals(M_base.index) or not M_rys.columns.equals(M_base.columns):
        raise ValueError("RYS and base CKA matrices must share index and columns.")
    diff = (M_rys - M_base).to_numpy()
    bound = float(max(0.05, abs(diff).max()))
    fig = go.Figure(
        data=go.Heatmap(
            z=diff,
            x=list(M_base.columns),
            y=list(M_base.index),
            colorscale="RdBu",
            reversescale=True,
            zmin=-bound,
            zmax=bound,
            colorbar={"title": "ΔCKA"},
            hovertemplate=(
                "layer i = %{y}<br>layer j = %{x}<br>ΔCKA = %{z:+.3f}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(autorange="reversed", title="layer i")
    fig.update_xaxes(title="layer j")
    fig.update_layout(
        title=title,
        width=620,
        height=560,
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )
    if window is not None:
        s, e = window
        fig.add_shape(
            type="rect",
            x0=s - 0.5,
            x1=e - 0.5,
            y0=s - 0.5,
            y1=e - 0.5,
            line={"color": ACCENT, "width": 3},
            fillcolor="rgba(0,0,0,0)",
        )
    return fig
