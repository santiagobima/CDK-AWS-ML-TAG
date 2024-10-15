#!/usr/bin/env python3

import sys
import os
import aws_cdk as cdk
import aws_cdk.aws_ec2 as ec2  # Importamos ec2 para manejar la VPC

from stacks.sagemaker_stack import SagemakerStack
from stacks.pipeline_stack import PipelineStack

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

# Crear el stack de recursos de SageMaker (roles, buckets, etc.), pasando el nombre de la VPC
sagemaker_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc_name=vpc_name, env=env)

# Crear el stack de pipelines, si es necesario puedes pasar una fábrica aquí
pipeline_stack = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=None,  # Aquí puedes pasar la fábrica de pipeline si es necesario
    env=env
)

# Generar el template
app.synth()
