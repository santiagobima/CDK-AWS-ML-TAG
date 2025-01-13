# Usar una imagen base compatible
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /opt/ml/processing/input/code

# Copiar archivos necesarios
COPY simple_step.py .
COPY requirements.txt .

# Asegurar permisos de ejecución para el script
RUN chmod +x simple_step.py

# Instalar las dependencias requeridas
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Definir el comando por defecto
CMD ["python3", "simple_step.py"]
