FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean

# Copiar requirements y setup
COPY image/requirements.txt ./requirements.txt
COPY setup.py .

# Copiar el código fuente necesario desde el contexto raíz
COPY Pipelines/common/ pipelines/common/
COPY Pipelines/lead_conversion_rate/common/ pipelines/lead_conversion_rate/common/
COPY Pipelines/lead_conversion_rate/steps/ pipelines/lead_conversion_rate/steps/

# Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install .

# Para que Python imprima logs en tiempo real
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ENTRYPOINT fijo que ejecuta cualquier script pasado en el parámetro `code=...`
#ENTRYPOINT ["/bin/bash", "-c"]
#CMD ["exec python3 \"$0\" \"$@\""]
