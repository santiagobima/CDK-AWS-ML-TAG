FROM python:3.11-slim

# Copiar archivos necesarios
COPY simple_step.py /opt/ml/processing/input/code/simple_step.py
COPY requirements.txt /opt/ml/processing/input/code/requirements.txt

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r /opt/ml/processing/input/code/requirements.txt

# Comando por defecto
CMD ["python3", "/opt/ml/processing/input/code/simple_step.py"]
