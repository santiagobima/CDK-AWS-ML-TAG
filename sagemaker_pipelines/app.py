#!/usr/bin/env python3

import aws_cdk as cdk
from pipelines.definitions.lead_conversion_pipeline import LeadConversionFactory
from pipeline_stack import PipelineStack
import os 

app = cdk.App()

LOGICAL_PREFIX = "DSM"

# Verificar si ejecutar en modo local
local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

# Crear la pila del pipeline
lead_conversion_pipeline_stack = PipelineStack(
    app,
    id=f"{LOGICAL_PREFIX}-PipelinesStack",
    factory=LeadConversionFactory(pipeline_config_parameter="your_value_here")
)

app.synth()