FROM public.ecr.aws/lambda/python:3.11

# Necesario para modelos con LightGBM o sklearn
RUN yum install -y libgomp

# Instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar los archivos correctos de Lambda
COPY lambda_inference.py ./lambda_inference.py
COPY lambda_utils.py ./lambda_utils.py

# Handler principal
CMD ["lambda_inference.lambda_handler"]