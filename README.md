# ai-master-project

# UI Web — Mantenimiento Predictivo (Clasificación)

UI en **Gradio** (y **Streamlit** opcional) para cargar datasets, preprocesar, predecir, evaluar y explorar alarmas. Stubs/TODOs para conectar tu pipeline de Google Colab.

## Estructura del proyecto
.
├── app.py
├── predictor.py
├── viz.py
├── utils.py
├── streamlit_app.py         # (opcional)
├── requirements.txt
├── Dockerfile
├── Procfile                 # (opcional, si usaras otro PaaS)
├── README.md
└── diagrams.txt             # Diagrama ASCII del flujo

## Características
- Carga `.csv` / `.parquet`
- Mapeo de columnas (`id_equipo`, `fecha`, `target`, `features`, `instruccion_mantenimiento`, `alarma`)
- Preprocesado (stub)
- Predicción (stub) con contrato `y_pred` (+ `y_proba` si aplica)
- Métricas: accuracy, precision, recall, F1, ROC AUC, matriz de confusión, `classification_report`
- Visualizaciones (Plotly): ROC, CM, tabla HTML, `px.scatter` de alarmas
- Exportación de predicciones a CSV
- Logging a `app.log` y visor en UI
- Datos sintéticos de prueba (sin subir dataset)

## Integración con tu notebook
Edita `predictor.py`:
- `load_model()` **TODO**: carga tu modelo/pipeline real.
- `preprocess(df, mappings, model)` **TODO**: aplica tu preprocesado.
- `predict(df, mappings, model)` **TODO**: devuelve `y_pred` (y `y_proba` opcional).
- `evaluate(...)`: ya implementado (usa scikit-learn).

> La UI incluye un **test de contrato** para verificar que `predict()` devuelve las columnas esperadas.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py  # abrirá Gradio; añade --no-share si no quieres URL pública