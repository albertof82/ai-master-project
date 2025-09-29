#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
UI web (Gradio) para mantenimiento predictivo (clasificación falla/no falla).
- Carga CSV/Parquet
- Mapeo de columnas clave
- Preprocesado (stub)
- Predicción (stub)
- Evaluación (accuracy, precision, recall, f1, ROC AUC)
- Visualizaciones (tabla, curva ROC, matriz de confusión, scatter de alarmas)
- Exportación CSV de predicciones
- Logging a app.log + panel de logs
- Tests rápidos para contrato de predict()

Autor: Tú :)
"""

from __future__ import annotations

import argparse
import io
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from predictor import (
    load_model,
    preprocess,
    predict,
    evaluate,
    required_prediction_columns,
    generate_synthetic_data,
)
from viz import (
    build_roc_curve_fig,
    build_confusion_matrix_fig,
    build_predictions_table_html,
    build_alarm_scatter_fig,
)
from utils import (
    setup_logging,
    logger,
    summarize_dataframe,
    validate_column_mappings,
    last_log_lines,
    save_dataframe_csv_temp,
)

# Config logging
setup_logging("app.log")


# ---- Estado global sencillo (Gradio State) ----
class AppState:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.proc_df: Optional[pd.DataFrame] = None
        self.preds_df: Optional[pd.DataFrame] = None
        self.mappings: Dict[str, Optional[str] | List[str]] = {
            "id_equipo": None,
            "fecha": None,
            "target": None,
            "features": [],
            "instruccion_mantenimiento": None,
            "alarma": None,
        }
        self.model: Any = None
        self.metrics: Optional[Dict[str, Any]] = None


STATE = AppState()


# ---- Helpers de UI ----
def _read_uploaded_file(file: gr.File) -> Tuple[str, pd.DataFrame]:
    """Lee CSV o Parquet según la extensión."""
    if file is None:
        raise ValueError("No se recibió archivo. Sube un .csv o .parquet")

    name = os.path.basename(file.name)
    ext = os.path.splitext(name)[1].lower()

    logger.info(f"Intentando cargar archivo: {name}")
    if ext == ".csv":
        df = pd.read_csv(file.name)
    elif ext == ".parquet":
        import pyarrow  # noqa: F401  (para asegurar dependencia)
        df = pd.read_parquet(file.name)
    else:
        raise ValueError(f"Extensión no soportada: {ext}. Usa .csv o .parquet")

    if df.empty:
        raise ValueError("El archivo parece estar vacío.")

    return name, df


def _auto_guess_mappings(df: pd.DataFrame) -> Dict[str, Optional[str] | List[str]]:
    """Adivina columnas comunes para comodidad."""
    cols_lower = {c.lower(): c for c in df.columns}
    guess = {
        "id_equipo": cols_lower.get("equipo") or cols_lower.get("id_equipo"),
        "fecha": cols_lower.get("fecha") or cols_lower.get("timestamp") or cols_lower.get("ts"),
        "target": cols_lower.get("target") or cols_lower.get("label") or cols_lower.get("alarma"),
        "features": [c for c in df.columns if c not in {"fecha", "equipo", "id_equipo", "target", "label", "alarma", "instruccion_mantenimiento"}],
        "instruccion_mantenimiento": cols_lower.get("instruccion_mantenimiento"),
        "alarma": cols_lower.get("alarma"),
    }
    return guess


# ---- Callbacks ----
def cb_load_file(file: gr.File, use_synthetic: bool) -> Tuple[str, str, gr.Dataframe, List[str]]:
    """
    Carga dataset subido o genera datos sintéticos para demo.
    Devuelve: mensaje, resumen, head, columnas.
    """
    try:
        if use_synthetic:
            df = generate_synthetic_data(n=800)
            msg = "Se generaron datos sintéticos para prueba."
            filename = "synthetic_memory_df"
        else:
            filename, df = _read_uploaded_file(file)
            msg = f"Archivo '{filename}' cargado correctamente."

        STATE.raw_df = df.copy()
        STATE.proc_df = None
        STATE.preds_df = None
        STATE.metrics = None

        # Intento de mapeo automático
        STATE.mappings = _auto_guess_mappings(df)

        summary = summarize_dataframe(df)
        head = df.head(20)
        cols = list(df.columns)
        logger.info(f"Carga ok: shape={df.shape}")

        return msg, summary, head, cols
    except Exception as e:
        logger.exception("Error al cargar archivo")
        return f"❌ Error: {e}", "", pd.DataFrame(), []


def cb_set_mappings(
    id_equipo: Optional[str],
    fecha: Optional[str],
    target: Optional[str],
    features: List[str],
    instruccion: Optional[str],
    alarma: Optional[str],
) -> str:
    """Actualiza el mapeo de columnas desde la UI."""
    try:
        if STATE.raw_df is None:
            return "❌ Sube primero un dataset."

        STATE.mappings.update(
            {
                "id_equipo": id_equipo or None,
                "fecha": fecha or None,
                "target": target or None,
                "features": features or [],
                "instruccion_mantenimiento": instruccion or None,
                "alarma": alarma or None,
            }
        )
        validate_column_mappings(STATE.raw_df, STATE.mappings)
        logger.info(f"Nuevos mapeos: {STATE.mappings}")
        return "✅ Mapeos de columnas actualizados."
    except Exception as e:
        logger.exception("Error al actualizar mapeos")
        return f"❌ Error en mapeos: {e}"


def cb_preprocess() -> Tuple[str, gr.Dataframe, str]:
    """Ejecuta el preprocesado (stub delega en predictor.preprocess)."""
    try:
        if STATE.raw_df is None:
            return "❌ Sube primero un dataset.", pd.DataFrame(), ""

        validate_column_mappings(STATE.raw_df, STATE.mappings)
        # Carga/instancia el modelo (o pipeline) si aplica
        if STATE.model is None:
            STATE.model = load_model()  # TODO: conecta tu modelo real

        proc_df, prep_summary = preprocess(STATE.raw_df, STATE.mappings, model=STATE.model)
        STATE.proc_df = proc_df
        logger.info("Preprocesado completado.")
        return "✅ Preprocesado listo.", proc_df.head(20), prep_summary
    except Exception as e:
        logger.exception("Error en preprocesado")
        return f"❌ Error: {e}", pd.DataFrame(), traceback.format_exc()


def cb_predict(n_preview: int) -> Tuple[str, gr.Dataframe, str, gr.File]:
    """Ejecuta la predicción y ofrece descarga de CSV."""
    try:
        if STATE.proc_df is None:
            return "❌ Preprocesa el dataset primero.", pd.DataFrame(), "", None

        validate_column_mappings(STATE.proc_df, STATE.mappings, allow_missing_target=True)
        preds_df = predict(STATE.proc_df, STATE.mappings, model=STATE.model)  # TODO: conecta
        # Test rápido: contrato de columnas
        missing_cols = [c for c in required_prediction_columns() if c not in preds_df.columns]
        assert not missing_cols, f"predict() debe devolver columnas: {missing_cols}"

        STATE.preds_df = preds_df.copy()
        preview = preds_df.head(int(n_preview))

        # Guardar CSV temporal para descarga
        csv_path = save_dataframe_csv_temp(preds_df, filename="predicciones.csv")
        msg = f"✅ Predicción completada. Registros: {len(preds_df)}. CSV listo para descargar."
        logger.info("Predicción completada.")
        return msg, preview, f"Archivo guardado en: {csv_path}", csv_path
    except Exception as e:
        logger.exception("Error en predicción")
        return f"❌ Error: {e}", pd.DataFrame(), traceback.format_exc(), None


def cb_evaluate() -> Tuple[str, Dict[str, Any], str, object, object]:
    """
    Calcula métricas y devuelve:
    - Mensaje
    - Dict métricas
    - classification_report (texto)
    - Figura ROC (plotly) si aplica
    - Figura matriz de confusión (plotly)
    """
    try:
        if STATE.preds_df is None:
            return "❌ Genera predicciones primero.", {}, "", None, None

        y_true_col = STATE.mappings.get("target")
        if not y_true_col or y_true_col not in STATE.preds_df.columns:
            return "ℹ️ No hay columna de target en el dataset. Evaluación deshabilitada.", {}, "", None, None

        metrics = evaluate(STATE.preds_df, y_true_col=y_true_col, y_pred_col="y_pred")
        STATE.metrics = metrics

        roc_fig = None
        if metrics.get("roc_curve") is not None:
            fpr, tpr, thresholds, auc_value = metrics["roc_curve"]
            roc_fig = build_roc_curve_fig(fpr, tpr, auc_value)

        cm_fig = None
        if metrics.get("confusion_matrix") is not None:
            cm = metrics["confusion_matrix"]
            labels = metrics.get("confusion_labels", ["0", "1"])
            cm_fig = build_confusion_matrix_fig(cm, labels)

        cls_rep = metrics.get("classification_report_text", "")
        return "✅ Evaluación lista.", metrics, cls_rep, roc_fig, cm_fig
    except Exception as e:
        logger.exception("Error en evaluación")
        return f"❌ Error: {e}", {}, "", None, None


def cb_visualize_predictions_table(max_rows: int) -> str:
    """Tabla HTML de predicciones para preview."""
    try:
        if STATE.preds_df is None:
            return "<b>Sube y predice primero</b>"
        html = build_predictions_table_html(STATE.preds_df.head(int(max_rows)))
        return html
    except Exception as e:
        logger.exception("Error creando tabla")
        return f"<b>❌ Error:</b> {e}"


def cb_alarm_scatter(
    x_col: str,
    y_col: str,
    color_col: str,
    alarma_filter: Optional[str],
) -> object:
    """Gráfico px.scatter interactivo con filtro de alarma."""
    try:
        df = STATE.raw_df if STATE.preds_df is None else STATE.preds_df
        if df is None or df.empty:
            raise ValueError("No hay datos para graficar.")
        fig = build_alarm_scatter_fig(
            df=df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            alarma_col=STATE.mappings.get("alarma"),
            alarma_filter=alarma_filter,
        )
        return fig
    except Exception as e:
        logger.exception("Error en scatter de alarmas")
        return None


def cb_read_logs(n_lines: int) -> str:
    return last_log_lines("app.log", n=int(n_lines))


def cb_test_contract() -> str:
    """Test rápido del contrato de predict()."""
    try:
        if STATE.proc_df is None:
            return "❌ Preprocesa primero."
        preds_df = predict(STATE.proc_df, STATE.mappings, model=STATE.model)
        missing = [c for c in required_prediction_columns() if c not in preds_df.columns]
        if missing:
            return f"❌ predict() NO cumple. Faltan columnas: {missing}"
        return "✅ predict() cumple contrato: y_pred (+ y_proba si aplica)."
    except Exception as e:
        return f"❌ Error: {e}"


# ---- Construcción de la UI (Gradio Blocks) ----
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Mantenimiento Predictivo - Clasificación (Gradio)") as demo:
        gr.Markdown("# 🛠️ Mantenimiento Predictivo — Clasificación (UI Gradio)")
        gr.Markdown(
            "Carga tu dataset, mapea columnas, preprocesa, predice, evalúa y explora alarmas. "
            "**Nota**: conecta tus funciones reales en `predictor.py` (marcadas con TODO)."
        )

        with gr.Tabs():
            # --- Tab: Upload ---
            with gr.Tab("1) Cargar"):
                with gr.Row():
                    file = gr.File(label="Sube archivo (.csv o .parquet)", file_count="single", type="file")
                    use_synth = gr.Checkbox(label="Usar datos sintéticos de ejemplo", value=False)
                load_btn = gr.Button("Cargar")
                load_msg = gr.Markdown()
                df_summary = gr.Markdown(label="Resumen del dataset")
                df_head = gr.Dataframe(interactive=False, label="Vista preliminar (primeras filas)")

                cols = gr.State([])
                load_btn.click(cb_load_file, inputs=[file, use_synth], outputs=[load_msg, df_summary, df_head, cols])

            # --- Tab: Mapeo columnas ---
            with gr.Tab("2) Mapeo columnas"):
                gr.Markdown("Selecciona las columnas correspondientes. Se intentan adivinar automáticamente.")
                id_equipo = gr.Dropdown(label="id_equipo / equipo", choices=[], interactive=True)
                fecha = gr.Dropdown(label="fecha / timestamp", choices=[], interactive=True)
                target = gr.Dropdown(label="target (si existe)", choices=[], interactive=True)
                features = gr.Dropdown(label="features (multi)", choices=[], multiselect=True, interactive=True)
                instruccion = gr.Dropdown(label="instruccion_mantenimiento (opcional)", choices=[], interactive=True)
                alarma = gr.Dropdown(label="alarma (opcional)", choices=[], interactive=True)

                set_map_btn = gr.Button("Guardar mapeos")
                map_msg = gr.Markdown()

                # Sincroniza choices tras carga
                def _fill_choices(c):
                    return c, c, c, c, c, c

                load_btn.click(_fill_choices, inputs=[cols], outputs=[id_equipo, fecha, target, features, instruccion, alarma])

                # Set valores sugeridos / auto-guess
                def _defaults():
                    m = STATE.mappings
                    return m.get("id_equipo"), m.get("fecha"), m.get("target"), m.get("features"), m.get("instruccion_mantenimiento"), m.get("alarma")

                load_btn.click(_defaults, outputs=[id_equipo, fecha, target, features, instruccion, alarma])

                set_map_btn.click(
                    cb_set_mappings,
                    inputs=[id_equipo, fecha, target, features, instruccion, alarma],
                    outputs=[map_msg],
                )

            # --- Tab: Preprocesar ---
            with gr.Tab("3) Preprocesar"):
                prep_btn = gr.Button("Preprocesar")
                prep_msg = gr.Markdown()
                prep_head = gr.Dataframe(interactive=False, label="Post-preprocesado (vista)")
                prep_summary = gr.Markdown(label="Resumen del preprocesado")
                prep_btn.click(cb_preprocess, outputs=[prep_msg, prep_head, prep_summary])

            # --- Tab: Predecir / Exportar ---
            with gr.Tab("4) Predecir / Exportar"):
                n_preview = gr.Number(label="Filas a previsualizar", value=20, precision=0)
                pred_btn = gr.Button("Predecir")
                pred_msg = gr.Markdown()
                pred_head = gr.Dataframe(interactive=False, label="Predicciones (preview)")
                pred_path_msg = gr.Markdown()
                pred_download = gr.File(label="Descargar predicciones.csv", interactive=False)

                pred_btn.click(cb_predict, inputs=[n_preview], outputs=[pred_msg, pred_head, pred_path_msg, pred_download])

            # --- Tab: Evaluar ---
            with gr.Tab("5) Evaluar"):
                eval_btn = gr.Button("Calcular métricas")
                eval_msg = gr.Markdown()
                metrics_json = gr.JSON(label="Métricas")
                cls_report = gr.Textbox(label="classification_report", interactive=False, lines=12)
                roc_plot = gr.Plot(label="Curva ROC")
                cm_plot = gr.Plot(label="Matriz de confusión")
                eval_btn.click(cb_evaluate, outputs=[eval_msg, metrics_json, cls_report, roc_plot, cm_plot])

            # --- Tab: Visualizaciones ---
            with gr.Tab("6) Visualizar resultados"):
                gr.Markdown("Vista de tabla de predicciones (HTML) y otras visualizaciones.")
                max_rows = gr.Number(label="Máx. filas en tabla HTML", value=100, precision=0)
                table_btn = gr.Button("Mostrar tabla")
                table_html = gr.HTML()
                table_btn.click(cb_visualize_predictions_table, inputs=[max_rows], outputs=[table_html])

            # --- Tab: Alarmas (px.scatter) ---
            with gr.Tab("7) Explorar alarmas"):
                gr.Markdown("Gráfico interactivo `px.scatter` configurable.")
                x_col = gr.Dropdown(label="Eje X (por defecto: fecha)", choices=[], interactive=True)
                y_col = gr.Dropdown(label="Eje Y (por defecto: equipo)", choices=[], interactive=True)
                color_col = gr.Dropdown(label="Color (por defecto: instruccion_mantenimiento)", choices=[], interactive=True)
                alarma_filter = gr.Textbox(label="Filtro exacto por 'alarma' (opcional)")
                scatter_btn = gr.Button("Dibujar scatter")
                scatter_plot = gr.Plot(label="Alarmas — scatter interactivo")

                # Popular choices con columnas dataset
                load_btn.click(_fill_choices, inputs=[cols], outputs=[x_col, y_col, color_col, gr.State(None), gr.State(None), gr.State(None)])

                # Defaults sugeridos
                def _alarm_defaults():
                    m = STATE.mappings
                    return (m.get("fecha"), m.get("id_equipo") or "equipo", m.get("instruccion_mantenimiento"))

                load_btn.click(_alarm_defaults, outputs=[x_col, y_col, color_col])

                scatter_btn.click(cb_alarm_scatter, inputs=[x_col, y_col, color_col, alarma_filter], outputs=[scatter_plot])

            # --- Tab: Logs ---
            with gr.Tab("8) Logs"):
                n_lines = gr.Number(label="Últimas N líneas", value=200, precision=0)
                show_logs_btn = gr.Button("Mostrar logs")
                logs_view = gr.Textbox(label="app.log (tail)", lines=20)
                show_logs_btn.click(cb_read_logs, inputs=[n_lines], outputs=[logs_view])

            # --- Tab: Tests rápidos ---
            with gr.Tab("9) Tests"):
                test_btn = gr.Button("Test contrato predict()")
                test_msg = gr.Markdown()
                test_btn.click(cb_test_contract, outputs=[test_msg])

        gr.Markdown(
            "—\n**Sugerencia Colab**: ejecuta `demo.launch(share=True)` para obtener URL pública automáticamente."
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--no-share", action="store_true", help="No abrir túnel público (útil local/producción).")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue()  # feedback/spinners
    demo.launch(server_name=args.host, server_port=args.port, share=not args.no_share)


if __name__ == "__main__":
    main()
