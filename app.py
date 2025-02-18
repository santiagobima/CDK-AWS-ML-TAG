#!/usr/bin/env python3

import os
import logging
import aws_cdk as cdk
from dotenv import load_dotenv
from constructs import Construct

from Stack.sagemaker_stack import SagemakerStack
from Stack.pipeline_stack import PipelineStack
from Pipelines.lead_conversion_rate.definition import LeadConversionFactory

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOGICAL_PREFIX = "DSM"

# Definir el modo de ejecución local aquí en el "control room"
LOCAL_MODE = False  # Cambiar a True para ejecutar el cálculo en modo local

def load_environment(app):
    """
    Carga el archivo .env según el entorno especificado en `cdk deploy --context env=...`.
    """
    env = app.node.try_get_context('env') or 'dev'
    if not env:
        raise ValueError("El entorno no está especificado. Use `--context env=dev` o `--context env=prod`.")
    
    env_file = f"env/.env.{env}"
    load_dotenv(env_file)
    logger.info(f"Entorno seleccionado: {env}")
    logger.info(f"Archivo de entorno cargado: {env_file}")
    logger.info(f"DATA_BUCKET: {os.getenv('DATA_BUCKET')}")
    logger.info(f"SOURCES_BUCKET: {os.getenv('SOURCES_BUCKET')}")

# Crear la aplicación CDK
app = cdk.App()

# Cargar las variables de entorno a partir del contexto
load_environment(app)

SOURCE_BUCKET = os.getenv("SOURCES_BUCKET")
assert SOURCE_BUCKET is not None

# Configuración de entorno y VPC desde variables de entorno
account = os.getenv('CDK_DEFAULT_ACCOUNT')
region = os.getenv('CDK_DEFAULT_REGION')
vpc_name = os.getenv("VPC_ID")

if not account or not region or not vpc_name:
    raise ValueError("Faltan variables de entorno necesarias: CDK_DEFAULT_ACCOUNT, CDK_DEFAULT_REGION o VPC_ID.")

# Crear el stack de recursos de SageMaker

# Obtener el prefijo del contexto de CDK
LOGICAL_PREFIX = app.node.try_get_context("resource_prefix")
if not LOGICAL_PREFIX:
    raise ValueError("El prefijo lógico (resource_prefix) no está definido en cdk.json")

sagemaker_stack = SagemakerStack(
    app, 
    id=f"{LOGICAL_PREFIX}-SagemakerStack",
    env=cdk.Environment(account=account, region=region),
    vpc_name=vpc_name,
    local_mode=LOCAL_MODE  # Pasamos `local_mode` desde app.py
)
logger.info("Stack de SageMaker configurado correctamente.")

# Crear el stack de pipelines y establecer la dependencia en el stack de SageMaker

print(f"DEBUG - ARN del rol de SageMaker que se pasa al pipeline: {sagemaker_stack.sm_execution_role.role_arn}")

lead_conversion_pipeline = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=LeadConversionFactory(local_mode=LOCAL_MODE),
    env=cdk.Environment(account=account, region=region),
    local_mode=LOCAL_MODE,  # Pasamos `local_mode` desde app.py
    pipeline_name="LeadConversionPipeline",
    source_bucket_name=SOURCE_BUCKET,
    sm_execution_role_arn=sagemaker_stack.sm_execution_role.role_arn,
)
lead_conversion_pipeline.add_dependency(sagemaker_stack)
logger.info("Stack de pipelines configurado correctamente con dependencia en el stack de SageMaker.")

# Generar el template de CDK
app.synth()
