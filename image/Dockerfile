FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias necesarias para compilar paquetes como psutil
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean

# Copiar requirements
COPY image/requirements.txt ./requirements.txt

# Copiar todo lo necesario desde el contexto raíz (el build context de CDK)
COPY setup.py .
COPY Pipelines/common/ pipelines/common/
COPY Pipelines/lead_conversion_rate/common/ pipelines/lead_conversion_rate/common/
COPY Pipelines/lead_conversion_rate/steps/ pipelines/lead_conversion_rate/steps/

# Instalar dependencias
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt && pip install .

ENV PYTHONUNBUFFERED=1

# Entrypoint dinámico para que SageMaker pueda ejecutar cualquier script
ENTRYPOINT ["python3"]