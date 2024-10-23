#!/usr/bin/env python3

import aws_cdk as cdk
import aws_cdk.aws_ec2 as ec2
from constructs import Construct
import os

from stacks.sagemaker_stack import SagemakerStack
from stacks.pipeline_stack import PipelineStack
from pipelines.definitions.lead_conversion_pipeline import LeadConversionFactory  # Importamos la fábrica

LOGICAL_PREFIX = "DSM"

# Función para obtener el entorno de la cuenta y la región
def get_environment_from_context(app):
    account = app.node.try_get_context("account")
    region = app.node.try_get_context("region")
    return cdk.Environment(account=account, region=region)

# Crear la aplicación CDK
app = cdk.App()

# Obtener el entorno y el nombre de la VPC del contexto
env = get_environment_from_context(app)
vpc_name = app.node.try_get_context("vpc_name")

# Validar que el nombre de la VPC esté presente en el contexto
if not vpc_name:
    raise ValueError("The VPC name was not found in the context. Please specify 'vpc_name' in cdk.json.")

# Crear la instancia de la fábrica de pipelines
factory = LeadConversionFactory(pipeline_config_parameter="Cloud Developer")

# Crear el stack de recursos de SageMaker, pasando el nombre de la VPC
sagemaker_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc_name=vpc_name, env=env)

# Crear el stack de pipelines, pasando la fábrica válida
pipeline_stack = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=factory,  # Pasa la fábrica aquí
    env=env
)

# Generar el template
app.synth()

