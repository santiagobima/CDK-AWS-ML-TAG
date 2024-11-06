import os
import sagemaker
import sagemaker.image_uris
from sagemaker import LocalSession, ScriptProcessor
from sagemaker.workflow import parameters
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from pipelines.definitions.base import SagemakerPipelineFactory

class LeadConversionFactory(SagemakerPipelineFactory):
    pipeline_config_parameter: str

    def create(
        self,
        role: str,
        pipeline_name: str,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        # Determinar si se está en modo local
        local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"

        # Parámetro de instancia (local o cloud)
        instance_type_var = parameters.ParameterString(
            name="InstanceType",
            default_value="local" if isinstance(sm_session, LocalSession) else "ml.m5.large"
        )

        # Configurar el URI de la imagen de SKLearn proporcionada por AWS SageMaker
        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=sm_session.boto_region_name,
            version="0.23-1",
        )

        # Crear el ScriptProcessor con la configuración adecuada
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type_var,
            instance_count=1,
            role=role,
            sagemaker_session=sm_session,
        )

        # Obtener el bucket de datos desde la variable de entorno, y eliminar cualquier barra al final
        data_bucket_name = os.getenv("DATA_BUCKET").rstrip('/')

        # Configurar los inputs y outputs dependiendo del modo
        if local_mode:
            code_path = os.path.abspath("./pipelines/sources/lead_conversion/evaluate.py")
            inputs = []
            outputs = []
        else:
            code_path = "pipelines/sources/lead_conversion/evaluate.py"

            # Definir inputs y outputs basados en S3 usando el bucket de datos del entorno
            inputs = [
                sagemaker.processing.ProcessingInput(
                    source=f"s3://{data_bucket_name}/input-data",  # Ajuste para una subcarpeta si es necesario
                    destination="/opt/ml/processing/input"
                )
            ]
            outputs = [
                sagemaker.processing.ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{data_bucket_name}/output-data"
                )
            ]

        # Crear el paso de procesamiento
        processing_step = ProcessingStep(
            name="processing-example",
            step_args=processor.run(
                code=code_path,
                inputs=inputs,
                outputs=outputs
            ),
            job_arguments=[
                "--config-parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )

        # Paso 2: Paso de procesamiento local sin S3
        processing_step_2 = ProcessingStep(
            name="local-processing-step",
            step_args=processor.run(
                code="pipelines/sources/lead_conversion/simple_step.py",
                inputs=[],
                outputs=[],
            ),
        )

        # Definir los pasos del pipeline en función del modo
        if local_mode:
            steps = [processing_step_2]
        else:
            steps = [processing_step]

        # Definir el pipeline completo
        return Pipeline(
            name=pipeline_name,
            steps=steps,
            sagemaker_session=sm_session,
            parameters=[instance_type_var],
        )


"""This error is thrown because in SageMaker's ProcessingStep, either step_args or processor is required, but not both at the same time.In your lead_conversion_definition.py, you're defining processing_step_2 without the required arguments:"""
"""The ExamplePipeline class implements a specific SageMaker pipeline, inheriting from SagemakerPipelineFactory.
""It defines an instance type parameter that can be configured at runtime, depending on whether the session is local or cloud-based.
The pipeline uses the scikit-learn image provided by AWS to run a Python script (evaluate.py) in a ScriptProcessor.
A processing step is created using the ScriptProcessor, and the custom configuration parameter (pipeline_config_parameter) is passed as an argument to the script."""