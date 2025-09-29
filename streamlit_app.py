# -*- coding: utf-8 -*-
"""
streamlit_app.py
Variante Streamlit simple que reutiliza predictor.py y viz.py.
Pensada para correr local o en Colab con cloudflared/pyngrok.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from predictor import (
    load_model,
    preprocess,
    predict,
    evaluate,
    required_prediction_columns,
    generate_synthetic_data,
)
from viz import build_roc_curve_fig, build_confusion_matrix_fig, build_alarm_scatter_fig
from utils import summarize_dataframe, validate_column_mappings, save_dataframe_csv_temp


st.set_page_config(page_title="Mantenimiento Predictivo (Streamlit)", layout="wide")
st.title("🛠️ Mantenimiento Predictivo — Streamlit")

st.sidebar.header("Carga de datos")
use_synth = st.sidebar.checkbox("Usar datos sintéticos", value=False)
uploaded = st.sidebar.file_uploader("Sube CSV/Parquet", type=["csv", "parquet"])

if use_synth:
    df = generate_synthetic_data(n=800)
else:
    df = None
    if uploaded is not None:
        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_parquet(uploaded)

if df is None:
    st.info("Sube un dataset o activa 'Usar datos sintéticos'.")
    st.stop()

st.subheader("Resumen del dataset")
st.text(summarize_dataframe(df))
st.dataframe(df.head(20))

# Mapeo columnas
st.subheader("Mapeo de columnas")
cols = list(df.columns)
id_equipo = st.selectbox("id_equipo/equipo", [""] + cols, index=cols.index("equipo") + 1 if "equipo" in cols else 0)
fecha = st.selectbox("fecha/timestamp", [""] + cols, index=cols.index("fecha") + 1 if "fecha" in cols else 0)
target = st.selectbox("target (opcional)", [""] + cols, index=cols.index("alarma") + 1 if "alarma" in cols else 0)
features = st.multiselect("features", [c for c in cols if c not in {id_equipo, fecha, target}], default=[])
instruccion = st.selectbox("instruccion_mantenimiento (opcional)", [""] + cols, index=cols.index("instruccion_mantenimiento") + 1 if "instruccion_mantenimiento" in cols else 0)
alarma = st.selectbox("alarma (opcional)", [""] + cols, index=cols.index("alarma") + 1 if "alarma" in cols else 0)

mappings = {
    "id_equipo": id_equipo or None,
    "fecha": fecha or None,
    "target": target or None,
    "features": features or [],
    "instruccion_mantenimiento": instruccion or None,
    "alarma": alarma or None,
}
validate_column_mappings(df, mappings, allow_missing_target=True)

# Preprocesar
st.subheader("Preprocesar")
model = load_model()
proc_df, prep_summary = preprocess(df, mappings, model=model)
st.success("Preprocesado completado")
st.text(prep_summary)
st.dataframe(proc_df.head(20))

# Predecir
st.subheader("Predecir")
preds_df = predict(proc_df, mappings, model=model)
missing_cols = [c for c in required_prediction_columns() if c not in preds_df.columns]
assert not missing_cols, f"Faltan columnas en predict(): {missing_cols}"
st.dataframe(preds_df.head(20))
csv_path = save_dataframe_csv_temp(preds_df, "predicciones.csv")
st.download_button("Descargar predicciones.csv", data=open(csv_path, "rb"), file_name="predicciones.csv")

# Evaluar
st.subheader("Evaluar")
if mappings.get("target") and mappings["target"] in preds_df.columns:
    metrics = evaluate(preds_df, y_true_col=mappings["target"], y_pred_col="y_pred")
    st.json(metrics)
    if metrics.get("roc_curve"):
        fpr, tpr, thr, auc_value = metrics["roc_curve"]
        st.plotly_chart(build_roc_curve_fig(fpr, tpr, auc_value), use_container_width=True)
    if metrics.get("confusion_matrix"):
        st.plotly_chart(build_confusion_matrix_fig(metrics["confusion_matrix"], metrics.get("confusion_labels", ["0","1"])), use_container_width=True)
else:
    st.info("No hay columna target en el dataset. Evaluación deshabilitada.")

# Alarmas
st.subheader("Explorar alarmas (px.scatter)")
x_col = st.selectbox("X", [fecha] + cols if fecha else cols, index=0 if fecha else 0)
y_col = st.selectbox("Y", [id_equipo] + cols if id_equipo else cols, index=0 if id_equipo else 0)
color_col = st.selectbox("Color", [instruccion] + cols if instruccion else cols, index=0 if instruccion else 0)
alarma_filter = st.text_input("Filtro exacto por 'alarma' (opcional)", "")
fig = build_alarm_scatter_fig(preds_df if preds_df is not None else df, x_col, y_col, color_col, mappings.get("alarma"), alarma_filter or None)
st.plotly_chart(fig, use_container_width=True)
