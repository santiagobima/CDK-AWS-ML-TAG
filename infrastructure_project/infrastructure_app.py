#!/usr/bin/env python3

import aws_cdk as cdk
from infrastructure_project.vpc_stack import VpcStack
from infrastructure_project.sagemaker_stack import SagemakerStack
import aws_cdk.aws_ec2 as ec2
from constructs import Construct

app = cdk.App()

LOGICAL_PREFIX = "DSM"

# Define the environment with account and region from context
account = app.node.try_get_context("account")
region = app.node.try_get_context("region")
env = cdk.Environment(account=account, region=region)

vpc_name = app.node.try_get_context("vpc_name")

class VpcStackWithLookup(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        if vpc_name:
            self.vpc = ec2.Vpc.from_lookup(self, id=f"{LOGICAL_PREFIX}-ImportedVpc", vpc_id=vpc_name)
        else:
            # Create new VPC
            vpc_stack = VpcStack(self, id=f"{LOGICAL_PREFIX}-VpcStack", env=env)
            self.vpc = vpc_stack.vpc

# Create the VPC stack and the SageMaker stack
vpc_stack_with_lookup = VpcStackWithLookup(app, id=f"{LOGICAL_PREFIX}-VpcStackLookup", env=env)

infra_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc=vpc_stack_with_lookup.vpc, env=env)

app.synth()
