#!/usr/bin/env python3

import aws_cdk as cdk
from infrastructure_project.vpc_stack import VpcStack
from infrastructure_project.sagemaker_stack import SagemakerStack
import aws_cdk.aws_ec2 as ec2

app = cdk.App()

LOGICAL_PREFIX = "DSM"

# Try to fetch the vpc_name from the context
vpc_name = app.node.try_get_context("vpc_name")

# If vpc_name is provided, use the existing VPC, else create a new VPC stack
if not vpc_name:
    vpc_stack = VpcStack(app, id=f"{LOGICAL_PREFIX}-VpcStack")
    vpc = vpc_stack.vpc
else:
    # Fetch the existing VPC by name
    vpc = ec2.Vpc.from_lookup(
        app, 
        id=f"{LOGICAL_PREFIX}-ImportedVpc", 
        vpc_name=vpc_name
    )
    vpc_stack = None

# Pass the VPC (either newly created or existing) to the SageMaker stack
infra_stack = SagemakerStack(app, id=f"{LOGICAL_PREFIX}-SagemakerStack", vpc=vpc)

# Add a dependency if a new VPC stack was created
if vpc_stack:
    infra_stack.add_dependency(vpc_stack)

app.synth()
