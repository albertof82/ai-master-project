# -*- coding: utf-8 -*-
"""
predictor.py
Lógica de modelo/pipeline. Contiene stubs (TODO) para conectar tu notebook de Colab.
Incluye una implementación de fallback con datos sintéticos y un RandomForest de ejemplo
para que la UI sea usable sin tu código.

Funciones contrato:
- load_model() -> Any
- preprocess(df: pd.DataFrame, mappings: Dict[str, Any], model: Any) -> Tuple[pd.DataFrame, str]
- predict(df: pd.DataFrame, mappings: Dict[str, Any], model: Any) -> pd.DataFrame
- evaluate(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[str, Any]
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
os.environ["MODEL_PATH"]    = "/content/model.pkl"
os.environ["METADATA_PATH"] = "/content/metadata.json"



# ---- Helpers contrato ----
def required_prediction_columns() -> List[str]:
    """Columnas mínimas que debe devolver predict()."""
    return ["y_pred"]  # y_proba es recomendable si aplica


# ---- Datos sintéticos (para demo) ----
def generate_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Genera datos con columnas frecuentes: fecha, equipo, instruccion_mantenimiento, alarma (target),
    + features numéricas/categóricas.
    """
    rng = np.random.default_rng(seed)
    equipos = [f"EQ-{i:02d}" for i in range(1, 16)]
    fechas = pd.date_range("2024-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "fecha": np.random.choice(fechas, size=n, replace=True),
        "equipo": rng.choice(equipos, size=n),
        "temp": rng.normal(60, 10, size=n).round(2),
        "vibracion": rng.normal(5, 1.2, size=n).round(2),
        "carga": rng.uniform(0.2, 0.95, size=n).round(3),
        "instruccion_mantenimiento": rng.choice(["Inspección", "Lubricar", "Reemplazar", "N/A"], size=n, p=[0.2,0.25,0.1,0.45]),
    })
    # Prob de alarma aumenta con temp/vibración alta + carga alta
    score = 0.015*(df["temp"]-55) + 0.2*(df["vibracion"]-4.5) + 0.8*(df["carga"]-0.6)
    prob = 1/(1+np.exp(-score))
    df["alarma"] = (prob > 0.55).astype(int)
    return df.sort_values("fecha").reset_index(drop=True)


# ---- STUBS/TODO: Conecta aquí tu pipeline real ----
def load_model() -> Any:
    """
    TODO: Carga/instancia tu modelo real (p.ej., RandomForestClassifier entrenado, o pipeline).
    Puedes cargar desde archivo pickle/Joblib o reconstruir desde tu notebook.
    Para demo, devolvemos un pipeline entrenado on-the-fly a partir de datos sintéticos.
    """
    # DEMO: pipeline simple con OHE para 'equipo' y 'instruccion_mantenimiento'
    demo_df = generate_synthetic_data(n=2000)
    feature_cols = ["temp", "vibracion", "carga", "equipo", "instruccion_mantenimiento"]
    X = demo_df[feature_cols]
    y = demo_df["alarma"].astype(int)

    cat_cols = ["equipo", "instruccion_mantenimiento"]
    num_cols = ["temp", "vibracion", "carga"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=7)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


def preprocess(df: pd.DataFrame, mappings: Dict[str, Any], model: Any = None) -> Tuple[pd.DataFrame, str]:
    """
    TODO: Implementa tu preprocesado real respetando el contrato de entrada/salida.
    Debe devolver (df_preprocesado, resumen_texto).
    """
    # DEMO: casting de fecha y ordenado; forward-fill simple; copia de columnas originales
    df2 = df.copy()
    fecha_col = mappings.get("fecha")
    if fecha_col and fecha_col in df2.columns:
        df2[fecha_col] = pd.to_datetime(df2[fecha_col], errors="coerce")
    df2 = df2.sort_values(by=[fecha_col] if fecha_col in df2.columns else df2.columns[0]).reset_index(drop=True)

    # Relleno simple de nulos numéricos
    numeric_cols = df2.select_dtypes(include=["int64", "float64", "int32", "float32", "Int64", "Float64"]).columns
    for c in numeric_cols:
        df2[c] = df2[c].fillna(df2[c].median())

    # Nulos categóricos
    cat_cols = df2.select_dtypes(include=["object", "string"]).columns
    for c in cat_cols:
        df2[c] = df2[c].fillna("desconocido")

    summary = []
    summary.append(f"Filas: {len(df2)}, Columnas: {len(df2.columns)}")
    n_null = int(df2.isna().sum().sum())
    summary.append(f"Nulos totales tras preprocesado: {n_null}")
    if fecha_col and fecha_col in df2.columns:
        summary.append(f"Rango de fechas: {df2[fecha_col].min()} — {df2[fecha_col].max()}")

    return df2, "\n".join(summary)


def _default_feature_columns(df: pd.DataFrame, mappings: Dict[str, Any]) -> List[str]:
    feats = [c for c in mappings.get("features", []) if c in df.columns]
    if feats:
        return feats
    # Fallback: auto infer
    excl = {mappings.get("target"), mappings.get("fecha"), mappings.get("id_equipo")}
    return [c for c in df.columns if c not in excl]


def predict(df: pd.DataFrame, mappings: Dict[str, Any], model: Any = None) -> pd.DataFrame:
    """
    TODO: Sustituye por tu lógica real. Debe devolver al menos una columna 'y_pred'.
    Opcional: 'y_proba' (probabilidad de clase positiva).
    """
    df2 = df.copy()
    features = _default_feature_columns(df2, mappings)

    # DEMO: si model tiene predict_proba, úsalo.
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(df2[features])[:, -1]
        y_pred = (y_proba >= 0.5).astype(int)
        df2["y_proba"] = y_proba
        df2["y_pred"] = y_pred
    elif hasattr(model, "predict"):
        y_pred = model.predict(df2[features])
        df2["y_pred"] = y_pred
    else:
        # Fallback tonto
        df2["y_pred"] = 0

    return df2


def evaluate(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[str, Any]:
    """
    Calcula métricas estándar. Si existe 'y_proba', computa ROC AUC y curva ROC.
    Devuelve un dict con métricas y artefactos (cm, cls_report, roc_curve points).
    """
    out: Dict[str, Any] = {}
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        return out

    y_true = df[y_true_col].astype(int)
    y_pred = df[y_pred_col].astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out.update(
        {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )

    # classification_report
    out["classification_report_text"] = classification_report(y_true, y_pred, digits=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    out["confusion_labels"] = ["0", "1"]

    # ROC AUC si hay y_proba
    if "y_proba" in df.columns:
        y_score = df["y_proba"].astype(float)
        try:
            auc_val = float(roc_auc_score(y_true, y_score))
            fpr, tpr, thr = roc_curve(y_true, y_score)
            out["roc_auc"] = auc_val
            out["roc_curve"] = (fpr.tolist(), tpr.tolist(), thr.tolist(), auc_val)
        except Exception:
            out["roc_auc"] = None
            out["roc_curve"] = None
    else:
        out["roc_auc"] = None
        out["roc_curve"] = None

    return out
