# Dockerfile

FROM python:3.8-slim

# Actualizar pip
RUN pip install --upgrade pip

# Instalar scikit-learn y otras dependencias necesarias
RUN pip install scikit-learn==1.3.0

# Establecer el directorio de trabajo
WORKDIR /opt/ml/processing/input/code/

# Comando por defecto
ENTRYPOINT ["python3"]

