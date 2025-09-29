# -*- coding: utf-8 -*-
"""
utils.py
Utilidades: logging, validación de columnas, resumen de DF, lectura de logs, guardado CSV temporal.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd


logger = logging.getLogger("mp_ui")


def setup_logging(log_path: str = "app.log", level: int = logging.INFO) -> None:
    """Configura logging a archivo y consola."""
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # Archivo
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # Evitar handlers duplicados
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logging inicializado")


def summarize_dataframe(df: pd.DataFrame) -> str:
    """Resumen compacto de shape, dtypes, nulos."""
    if df is None or df.empty:
        return "Dataset vacío."
    lines = [f"Shape: {df.shape[0]} filas × {df.shape[1]} columnas"]
    dtypes = df.dtypes.astype(str)
    lines.append("\n**Tipos de datos (primeras 20):**")
    lines.extend([f"- {c}: {t}" for c, t in dtypes.head(20).items()])
    nulls = df.isna().sum()
    n_null = int(nulls.sum())
    lines.append(f"\nNulos totales: {n_null}")
    if n_null > 0:
        lines.append("Columnas con nulos (top 10):")
        for c, v in nulls.sort_values(ascending=False).head(10).items():
            if v > 0:
                lines.append(f"- {c}: {int(v)}")
    return "\n".join(lines)


def validate_column_mappings(
    df: pd.DataFrame,
    mappings: Dict[str, Any],
    allow_missing_target: bool = False,
) -> None:
    """Valida que las columnas mapeadas existan."""
    required_keys = ["id_equipo", "fecha"]
    for k in required_keys:
        col = mappings.get(k)
        if col and col not in df.columns:
            raise ValueError(f"Columna mapeada '{k}'='{col}' no existe en el dataset.")

    tgt = mappings.get("target")
    if not allow_missing_target:
        if tgt and tgt not in df.columns:
            raise ValueError(f"Target mapeado '{tgt}' no existe en el dataset.")
    # features (si se han marcado)
    feats = mappings.get("features") or []
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Algunas features no existen en el dataset: {missing}")


def last_log_lines(path: str, n: int = 200) -> str:
    """Devuelve las últimas N líneas del log."""
    if not os.path.exists(path):
        return "(no hay logs aún)"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    tail = "".join(lines[-n:])
    return tail or "(sin contenido)"


def save_dataframe_csv_temp(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Guarda un CSV temporal (útil para gr.File)."""
    # En Colab / entornos temporales, /mnt/data suele estar disponible; si no, usar cwd.
    base_dir = "/mnt/data" if os.path.exists("/mnt/data") else "."
    path = os.path.join(base_dir, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"CSV guardado en {path} ({len(df)} filas)")
    return path
