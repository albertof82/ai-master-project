# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Evitar bytecode y buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema (pyarrow puede requerirlas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requisitos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Variables típicas en App Runner: PORT
ENV PORT=8080

# Exponer puerto (no estrictamente necesario en App Runner)
EXPOSE 8080

# Ejecutar Gradio vinculando host 0.0.0.0 y puerto desde $PORT
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080", "--no-share"]
