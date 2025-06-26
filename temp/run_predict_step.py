# temp/run_predict_step.py

import os
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import Session
from sagemaker.pytorch import PyTorchProcessor
from sagemaker.image_uris import retrieve

# Parámetros
role = "arn:aws:iam::415388300336:role/dsa-sm-execution-role"
bucket = os.environ.get("DATA_BUCKET", "datalake-tag")  # default de seguridad
region = boto3.Session().region_name
session = Session()

# Obtener la misma imagen que usás en tu pipeline
image_uri = retrieve(
    framework="pytorch",
    region=region,
    version="2.4.0",
    py_version="py311",
    image_scope="training",
    instance_type="ml.m5.4xlarge"
)

# Inicializar ScriptProcessor con la imagen gestionada
script_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.4xlarge",
    instance_count=1,
    base_job_name="predict-step",
    sagemaker_session=session
)

# Ejecutar el paso predict directamente
script_processor.run(
    code="pipelines/lead_conversion_rate/steps/predict.py",
    inputs=[
        ProcessingInput(source=f"s3://{bucket}/output-data", destination="/opt/ml/processing/predict_input_data"),
        ProcessingInput(source=f"s3://{bucket}/configs/model_config.yml", destination="/opt/ml/processing/configs"),
        ProcessingInput(source=f"s3://{bucket}/code/source_code", destination="/opt/ml/processing/source_code"),
        ProcessingInput(source=f"s3://{bucket}/code/source_code/summaries", destination="/opt/ml/processing/summaries")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/pickles/models",
                         destination=f"s3://{bucket}/output-data/predict/models"),
        ProcessingOutput(source="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/pickles/features",
                         destination=f"s3://{bucket}/output-data/predict/features"),
        ProcessingOutput(source="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/results",
                         destination=f"s3://{bucket}/output-data/predict/results")
    ]
)