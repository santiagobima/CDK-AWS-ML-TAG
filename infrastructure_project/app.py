#!/usr/bin/env python3

import aws_cdk as cdk
from infrastructure_project.vpc_stack import VpcStack
from infrastructure_project.sagemaker_stack import SagemakerStack
import aws_cdk.aws_ec2 as ec2
from constructs import Construct

LOGICAL_PREFIX = "DSM"


def get_environment_from_context(app):
    account = app.node.try_get_context("account") 
    region = app.node.try_get_context("region")
    return cdk.Environment(account=account, region=region)

# Función para obtener la VPC (importar o crear una nueva)
def get_vpc(scope, vpc_name, env):
    if vpc_name:
        return ec2.Vpc.from_lookup(scope, id=f"{LOGICAL_PREFIX}-ImportedVpc", vpc_id=vpc_name)
    else:
        # Crear nueva VPC
        vpc_stack = VpcStack(scope, id=f"{LOGICAL_PREFIX}-VpcStack", env=env)
        return vpc_stack.vpc

# Clase para manejar la VPC (con lookup o creación)
class VpcStackWithLookup(cdk.Stack):
    def __init__(self, scope: Construct, id: str, vpc_name: str, env: cdk.Environment, **kwargs) -> None:
        super().__init__(scope, id, env=env, **kwargs)  
        self.vpc = get_vpc(self, vpc_name, env)

# Crear la aplicación CDK
app = cdk.App()

# Obtener el entorno y nombre de la VPC del contexto
env = get_environment_from_context(app)
vpc_name = app.node.try_get_context("vpc_name")

# Crear la pila de la VPC (lookup o crear nueva)
vpc_stack_with_lookup = VpcStackWithLookup(app, id=f"{LOGICAL_PREFIX}-VpcStackLookup", vpc_name=vpc_name, env=env)

# Crear la pila de SageMaker utilizando la VPC creada o importada
infra_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc=vpc_stack_with_lookup.vpc, env=env)

# Generar el template
app.synth()


