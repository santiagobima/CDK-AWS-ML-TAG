FROM python:3.11-slim

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos desde `image/`
COPY requirements.txt .
#COPY *.py .

# Instalar dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Definir el comando por defecto
#CMD ["python3", "simple_step.py"]