# lead_conversion_pipeline.py

import sagemaker
from sagemaker import ScriptProcessor
from sagemaker.workflow import parameters
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline


class LeadConversionFactory:
    def __init__(self, pipeline_config_parameter: str):
        self.pipeline_config_parameter = pipeline_config_parameter

    def create(
        self,
        role: str,
        pipeline_name: str,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        # Definir un parámetro para configurar el tipo de instancia
        instance_type_var = parameters.ParameterString(
            name="InstanceType",
            default_value="local" if sm_session.local_mode else "ml.m5.large"
        )

        # Determinar el tipo de instancia y la imagen según el entorno
        if sm_session.local_mode:
            # Modo local
            image_uri = 'local-scikit-learn:1.3.0'
            actual_instance_type = 'local'
        else:
            # Modo en la nube
            image_uri = sagemaker.image_uris.retrieve(
                framework="sklearn",
                region=sm_session.boto_region_name,
                version="1.3-1",
                instance_type=instance_type_var,
            )
            actual_instance_type = instance_type_var

        # Crear un ScriptProcessor
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=actual_instance_type,
            instance_count=1,
            role=role,
            sagemaker_session=sm_session,
        )

        # Paso 1: Paso de procesamiento de ejemplo
        processing_step = ProcessingStep(
            name="processing-example",
            processor=processor,
            code="pipelines/sources/lead_conversion/evaluate.py",
            job_arguments=[
                "--config_parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )

        # Paso 2: Paso de procesamiento local sin S3
        processing_step_2 = ProcessingStep(
            name="local-processing-step",
            processor=processor,
            code="pipelines/sources/lead_conversion/simple_step.py",
        )

        # Definir el pipeline
        return Pipeline(
            name=pipeline_name,
            steps=[processing_step, processing_step_2],
            sagemaker_session=sm_session,
            parameters=[instance_type_var],
        )


"""This error is thrown because in SageMaker's ProcessingStep, either step_args or processor is required, but not both at the same time.In your lead_conversion_definition.py, you're defining processing_step_2 without the required arguments:"""
"""The ExamplePipeline class implements a specific SageMaker pipeline, inheriting from SagemakerPipelineFactory.
""It defines an instance type parameter that can be configured at runtime, depending on whether the session is local or cloud-based.
The pipeline uses the scikit-learn image provided by AWS to run a Python script (evaluate.py) in a ScriptProcessor.
A processing step is created using the ScriptProcessor, and the custom configuration parameter (pipeline_config_parameter) is passed as an argument to the script."""