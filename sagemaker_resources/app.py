#!/usr/bin/env python3

import aws_cdk as cdk
from sagemaker_stack import SagemakerStack
import aws_cdk.aws_ec2 as ec2
from constructs import Construct
import os
from dotenv import load_dotenv


LOGICAL_PREFIX = "DSM"

def get_environment_from_context(app):
    account = app.node.try_get_context("account") 
    region = app.node.try_get_context("region")
    return cdk.Environment(account=account, region=region)

# Función para obtener la VPC (importar una existente)
def get_vpc(scope, vpc_name, env):
    if vpc_name:
        return ec2.Vpc.from_lookup(scope, id=f"{LOGICAL_PREFIX}-ImportedVpc", vpc_id=vpc_name)
    else:
        raise ValueError("No VPC name provided in the context")

# Clase para manejar la VPC (solo lookup)
class VpcStackWithLookup(cdk.Stack):
    def __init__(self, scope: Construct, id: str, vpc_name: str, env: cdk.Environment, **kwargs) -> None:
        super().__init__(scope, id, env=env, **kwargs)  
        self.vpc = get_vpc(self, vpc_name, env)

# Crear la aplicación CDK
app = cdk.App()

# Obtener el entorno y nombre de la VPC del contexto
env = get_environment_from_context(app)
vpc_name = app.node.try_get_context("vpc_name")

if not vpc_name:
    raise ValueError("The VPC name was not found in the context. Please specify 'vpc_name' in cdk.json.")

# Crear la pila de la VPC (lookup)
vpc_stack_with_lookup = VpcStackWithLookup(app, id=f"{LOGICAL_PREFIX}-VpcStackLookup", vpc_name=vpc_name, env=env)

# Crear la pila de SageMaker utilizando la VPC importada
infra_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc=vpc_stack_with_lookup.vpc, env=env)

# Generar el template
app.synth()

