FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean

# Copiar requirements y setup
COPY image/requirements.txt ./requirements.txt
COPY setup.py .

# Copiar el código fuente necesario desde el contexto raíz (carpeta ahora es 'pipelines')
COPY pipelines/common/ pipelines/common/
COPY pipelines/lead_conversion_rate/common/ pipelines/lead_conversion_rate/common/
COPY pipelines/lead_conversion_rate/steps/ pipelines/lead_conversion_rate/steps/

# Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install .

# Para que Python imprima logs en tiempo real y reconozca la carpeta como paquete
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ENTRYPOINT flexible para ejecutar cualquier script pasado como parámetro
ENTRYPOINT ["python3"]

# ENTRYPOINT fijo que ejecuta cualquier script pasado en el parámetro `code=...`
#ENTRYPOINT ["/bin/bash", "-c"]
#CMD ["exec python3 \"$0\" \"$@\""]
