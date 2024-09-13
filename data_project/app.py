#!/usr/bin/env python3

import aws_cdk as cdk
from pipelines.definitions.example_pipeline_definition import LeadConversionFactory

from data_project.pipelines_stack import PipelineStack

app = cdk.App()

LOGICAL_PREFIX = "DSM"

lead_conversion_rate = PipelineStack(app, id=f"{LOGICAL_PREFIX}-PipelinesStack", factory=LeadConversionFactory())


app.synth()
