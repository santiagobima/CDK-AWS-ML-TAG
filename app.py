#!/usr/bin/env python3

import os
import logging
import aws_cdk as cdk
from dotenv import load_dotenv
from constructs import Construct
import boto3

from Stack.sagemaker_stack import SagemakerStack
from Stack.pipeline_stack import PipelineStack
from pipelines.lead_conversion_rate.definition import LeadConversionFactory

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOGICAL_PREFIX = "DSM"
LOCAL_MODE = False  # Cambiar a True si deseas ejecutar en local

def load_environment(app):
    env = app.node.try_get_context('env') or 'dev'
    if not env:
        raise ValueError("Debe usar --context env=dev o env=prod.")

    env_file = f"env/.env.{env}"
    load_dotenv(env_file)
    logger.info(f"Entorno seleccionado: {env}")
    logger.info(f"Archivo de entorno cargado: {env_file}")
    logger.info(f"DATA_BUCKET: {os.getenv('DATA_BUCKET')}")
    logger.info(f"SOURCES_BUCKET: {os.getenv('SOURCES_BUCKET')}")

# Crear aplicación CDK
app = cdk.App()
load_environment(app)

SOURCE_BUCKET = os.getenv("SOURCES_BUCKET")
assert SOURCE_BUCKET is not None

account = os.getenv('CDK_DEFAULT_ACCOUNT')
region = os.getenv('CDK_DEFAULT_REGION')
vpc_name = os.getenv("VPC_ID")

if not account or not region or not vpc_name:
    raise ValueError("Faltan variables: CDK_DEFAULT_ACCOUNT, CDK_DEFAULT_REGION o VPC_ID.")

LOGICAL_PREFIX = app.node.try_get_context("resource_prefix")
if not LOGICAL_PREFIX:
    raise ValueError("El prefijo lógico 'resource_prefix' no está definido en cdk.json")

# Crear stack de SageMaker
sagemaker_stack = SagemakerStack(
    app,
    id=f"{LOGICAL_PREFIX}-SagemakerStack",
    env=cdk.Environment(account=account, region=region),
    vpc_name=vpc_name,
    local_mode=LOCAL_MODE
)
logger.info("Stack de SageMaker creado.")

# Obtener ARN del rol desde SSM
ssm_client = boto3.client("ssm", region_name=region)
sm_execution_role_arn = ssm_client.get_parameter(
    Name=f"/{LOGICAL_PREFIX}/SagemakerExecutionRoleArn"
)["Parameter"]["Value"]

# Crear stack del pipeline
lead_conversion_pipeline = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=LeadConversionFactory(local_mode=LOCAL_MODE),
    env=cdk.Environment(account=account, region=region),
    local_mode=LOCAL_MODE,
    pipeline_name="LeadConversionPipeline-v2",
    source_bucket_name=SOURCE_BUCKET,
    sm_execution_role_arn=sm_execution_role_arn,
)

lead_conversion_pipeline.add_dependency(sagemaker_stack)
logger.info("Stack del pipeline creado correctamente.")

# Síntesis final
app.synth()