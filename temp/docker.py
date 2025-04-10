FROM python:3.11-slim

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos desde image/
COPY requirements.txt .
#COPY *.py .

# Instalar dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Definir el comando por defecto
#CMD ["python3", "simple_step.py"]


--------------


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
#COPY ../Pipelines/common/ pipelines/common/
#COPY ../Pipelines/lead_conversion_rate/common/ pipelines/lead_conversion_rate/common/
#COPY ../Pipelines/lead_conversion_rate/steps/ pipelines/lead_conversion_rate/steps/
COPY ../Pipelines pipelines/

# Instalar dependencias y tu paquete
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt && pip install .

# Evitar que Python haga buffering en logs
ENV PYTHONUNBUFFERED=1

# Este bloque permite que SageMaker llame cualquier script que le pases en `code=`
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["exec python3 \"$0\" \"$@\"", "pipelines/lead_conversion_rate/steps/simple_step.py"]


#ENTRYPOINT ["python3"]

-----

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
COPY ../Pipelines/common/ pipelines/common/
COPY ../Pipelines/lead_conversion_rate/common/ pipelines/lead_conversion_rate/common/
COPY ../Pipelines/lead_conversion_rate/steps/ pipelines/lead_conversion_rate/steps/


# Instalar dependencias y tu paquete
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt && pip install .

# Evitar que Python haga buffering en logs
ENV PYTHONUNBUFFERED=1

# Este bloque permite que SageMaker llame cualquier script que le pases en `code=`
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["exec python3 \"$0\" \"$@\""]
