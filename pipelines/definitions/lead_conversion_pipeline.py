import os
import sagemaker
import sagemaker.image_uris
from sagemaker import ScriptProcessor
from sagemaker.workflow import parameters
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from pipelines.definitions.base import SagemakerPipelineFactory
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LeadConversionFactory(SagemakerPipelineFactory):
    """
    Clase que define el pipeline de conversión de clientes potenciales.
    """

    pipeline_config_parameter: str
    local_mode: bool  # Configuración para determinar si el cálculo es en local

    def create(
        self,
        role: str,
        pipeline_name: str,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        """
        Crea el pipeline de SageMaker.

        :param role: ARN del rol de ejecución de SageMaker.
        :param pipeline_name: Nombre del pipeline.
        :param sm_session: Sesión de SageMaker.
        :return: Objeto `Pipeline` configurado.
        """
        # Determinar el tipo de instancia en función del modo
        instance_type_var = parameters.ParameterString(
            name="InstanceType",
            default_value="local" if self.local_mode else "ml.m5.large"
        )
        logger.info(f"Modo local: {self.local_mode}")

        # Configurar URI de imagen de SKLearn de AWS SageMaker
        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=sm_session.boto_region_name,
            version="0.23-1",
        )
        logger.info(f"URI de imagen SKLearn: {image_uri}")

        # Configurar ScriptProcessor
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type_var,
            instance_count=1,
            role=role,
            sagemaker_session=sm_session,
        )

        # Configurar inputs y outputs en S3 (sin cambios entre local y nube)
        data_bucket_name = os.getenv("DATA_BUCKET").rstrip('/')
        inputs, outputs = self._configure_io(data_bucket_name)

        # Crear pasos del pipeline
        processing_step = ProcessingStep(
            name="processing-example",
            step_args=processor.run(
                code="pipelines/sources/lead_conversion/evaluate.py",
                inputs=inputs,
                outputs=outputs,
            ),
            job_arguments=[
                "--config-parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )

        # Definir los pasos del pipeline
        steps = [processing_step]

        logger.info(f"Pipeline '{pipeline_name}' configurado con {len(steps)} paso(s).")

        # Retornar pipeline completo
        return Pipeline(
            name=pipeline_name,
            steps=steps,
            sagemaker_session=sm_session,
            parameters=[instance_type_var],
        )

    def _configure_io(self, data_bucket_name: str):
        """
        Configura inputs y outputs en S3 independientemente del modo.

        :param data_bucket_name: Nombre del bucket de datos.
        :return: Tupla con inputs y outputs configurados.
        """
        logger.info(f"Configurando pipeline con bucket S3 '{data_bucket_name}' para entrada y salida.")
        inputs = [
            sagemaker.processing.ProcessingInput(
                source=f"s3://{data_bucket_name}/input-data",
                destination="/opt/ml/processing/input"
            )
        ]
        outputs = [
            sagemaker.processing.ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{data_bucket_name}/output-data"
            )
        ]
        return inputs, outputs


"""This error is thrown because in SageMaker's ProcessingStep, either step_args or processor is required, but not both at the same time.In your lead_conversion_definition.py, you're defining processing_step_2 without the required arguments:"""
"""The ExamplePipeline class implements a specific SageMaker pipeline, inheriting from SagemakerPipelineFactory.
""It defines an instance type parameter that can be configured at runtime, depending on whether the session is local or cloud-based.
The pipeline uses the scikit-learn image provided by AWS to run a Python script (evaluate.py) in a ScriptProcessor.
A processing step is created using the ScriptProcessor, and the custom configuration parameter (pipeline_config_parameter) is passed as an argument to the script."""