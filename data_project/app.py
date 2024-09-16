#!/usr/bin/env python3

import aws_cdk as cdk
from pipelines.definitions.example_pipeline_definition import LeadConversionFactory
from data_project.pipelines_stack import PipelineStack

app = cdk.App()

LOGICAL_PREFIX = "DSM"

# Ensure you pass the required parameter 'pipeline_config_parameter' here
lead_conversion_rate = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=LeadConversionFactory(pipeline_config_parameter="your_value_here")
)

app.synth()
