# Dockerfile para Programador Quirúrgico Inteligente
FROM python:3.10-slim

# Metadatos
LABEL maintainer="tu@email.com"
LABEL version="4.9"
LABEL description="Sistema de optimización para programación quirúrgica"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY app/ ./app/
COPY notebooks/ ./notebooks/

# Exponer puerto
EXPOSE 7860

# Comando de inicio
CMD ["python", "app/programador_quirurgico_colab_v49.py"]
