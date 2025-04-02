FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Copiar archivos necesarios desde el contexto raíz
COPY ../setup.py .
COPY ../Pipelines/common/ Pipelines/common/

# Instalar la librería como paquete
RUN pip install --upgrade pip && pip install .

# Definir el comando por defecto (puede omitirse si no se necesita)
# CMD ["python3", "script.py"]