FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias necesarias para compilar paquetes como psutil
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean

# Copiar requirements
COPY ./image/requirements.txt ./requirements.txt

# Copiar archivos necesarios desde el contexto raíz
COPY ../setup.py .
COPY ../Pipelines/common/ Pipelines/common/

# Instalar dependencias y tu paquete
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install .

# CMD opcional
# CMD ["python3", "script.py"]