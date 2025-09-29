# -*- coding: utf-8 -*-
"""
viz.py
Gráficos y renderizaciones: ROC, matriz de confusión (plotly), tabla HTML, scatter de alarmas (px.scatter).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def build_roc_curve_fig(fpr: List[float], tpr: List[float], auc_value: Optional[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash")))
    fig.update_layout(
        title=f"Curva ROC (AUC={auc_value:.3f})" if auc_value is not None else "Curva ROC",
        xaxis_title="FPR",
        yaxis_title="TPR",
        template="plotly_white",
        height=420,
    )
    return fig


def build_confusion_matrix_fig(cm: Any, labels: List[str]) -> go.Figure:
    cm = np.array(cm)
    ztext = [[str(v) for v in row] for row in cm]
    fig = go.Figure(
        data=go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", showscale=True)
    )
    # anotar
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=labels[j], y=labels[i], text=ztext[i][j], showarrow=False, font=dict(color="black"))
    fig.update_layout(title="Matriz de confusión", xaxis_title="Predicho", yaxis_title="Real", template="plotly_white", height=420)
    return fig


def build_predictions_table_html(df: pd.DataFrame) -> str:
    """Render de tabla HTML simple para previsualizar predicciones."""
    # Evitar tablas gigantes
    html = df.to_html(index=False, justify="center")
    return f"<div style='max-height:500px; overflow:auto'>{html}</div>"


def build_alarm_scatter_fig(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str],
    alarma_col: Optional[str],
    alarma_filter: Optional[str],
):
    """Scatter interactivo para explorar alarmas."""
    dfx = df.copy()
    if alarma_col and alarma_filter:
        dfx = dfx[dfx[alarma_col].astype(str) == str(alarma_filter)]

    fig = px.scatter(
        dfx,
        x=x_col,
        y=y_col,
        color=color_col if color_col in dfx.columns else None,
        hover_data=dfx.columns,
        title="Eventos / Alarmas en el tiempo por equipo",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig
