import os
import sagemaker
import sagemaker.image_uris
from sagemaker import ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
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
        instance_type_var = ParameterString(
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

        # **Nuevo: agregar requirements.txt**
        requirements_input = sagemaker.processing.ProcessingInput(
            source=f"s3://{data_bucket_name}/requirements.txt",  # Cambiar por tu ruta
            destination="/opt/ml/processing/input/code/requirements.txt"
        )

        # Crear pasos del pipeline
        processing_step_1 = ProcessingStep(
            name="processing-evaluate",
            step_args=processor.run(
                code="pipelines/sources/lead_conversion/evaluate.py",
                inputs=[*inputs, requirements_input],
                outputs=outputs,
            ),
            job_arguments=[
                "--config-parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )
        
        processing_step_2 = ProcessingStep(
            name="processing-athena-query",
            step_args=processor.run(
                code="pipelines/sources/lead_conversion/athena_query.py",
                inputs=[*inputs, requirements_input],
                outputs=outputs,
            ),
        )

        # Definir los pasos del pipeline
        steps = [processing_step_1, processing_step_2]

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
