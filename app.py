#!/usr/bin/env python3

import os
import aws_cdk as cdk
from dotenv import load_dotenv
from constructs import Construct

from stacks.sagemaker_stack import SagemakerStack
from stacks.pipeline_stack import PipelineStack
from pipelines.definitions.lead_conversion_pipeline import LeadConversionFactory

LOGICAL_PREFIX = "DSM"

# Cargar el archivo .env en función del entorno
environment = os.getenv("ENVIRONMENT", "sandbox").lower()
env_file = f"env/.env.dev" if environment == "sandbox" else f"env/.env.prod"  
load_dotenv(env_file)

# Confirmación de variables de entorno
print(f"Entorno seleccionado: {environment}")
print(f"Archivo de entorno cargado: {env_file}")
print(f"DATA_BUCKET: {os.getenv('DATA_BUCKET')}")
print(f"SOURCES_BUCKET: {os.getenv('SOURCES_BUCKET')}")

def get_environment_from_context(app):
    account = app.node.try_get_context("account")
    region = app.node.try_get_context("region")
    return cdk.Environment(account=account, region=region)

# Crear la aplicación CDK
app = cdk.App()

# Obtener el entorno y el nombre de la VPC del contexto
env = get_environment_from_context(app)
vpc_name = app.node.try_get_context("vpc_name")

if not vpc_name:
    raise ValueError("The VPC name was not found in the context. Please specify 'vpc_name' in cdk.json.")

# Crear el stack de recursos de SageMaker
sagemaker_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc_name=vpc_name, env=env)

# Crear el stack de pipelines y establecer dependencia
lead_conversion_pipeline = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=LeadConversionFactory(pipeline_config_parameter="Cloud Developer"),
    env=env
)
lead_conversion_pipeline.add_dependency(sagemaker_stack)  # Establecer dependencia

# Generar el template
app.synth()
