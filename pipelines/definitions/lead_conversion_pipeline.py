import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from pipelines.definitions.base import SagemakerPipelineFactory
from pipelines.definitions.pipeline_step import PipelineStep
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
        scope,  # Se pasa el scope desde PipelineStack
        role: str,
        pipeline_name: str,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        """
        Crea el pipeline de SageMaker.

        :param scope: Alcance del constructo (proporcionado por PipelineStack).
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

        # Configurar inputs y outputs
        data_bucket_name = os.getenv("DATA_BUCKET").rstrip('/')
        inputs, outputs = self._configure_io(data_bucket_name)

        # Validar la ruta al archivo de código
        script_path = "pipelines/sources/lead_conversion/simple_step.py"
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"El archivo '{script_path}' no existe. Verifica la ruta.")

        # Configurar paso de preparación de datos con Docker
        data_prep_processor = PipelineStep(
            scope=scope,
            id="DataPrepProcessor",
            dockerfile_path="pipelines/sources/lead_conversion",
            step_name="data-preparation",
            command=["python3", "simple_step.py"],
            instance_type=instance_type_var,
            role=role,
            sagemaker_session=sm_session,
        ).create_processor()

        data_prep_step = ProcessingStep(
            name="DataPreparationStep",
            processor=data_prep_processor,
            inputs=inputs,
            outputs=outputs,
            code=script_path,
            job_arguments=[
                "--config-parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )

        # Configurar paso de inferencia con Docker
        inference_processor = PipelineStep(
            scope=scope,
            id="InferenceProcessor",
            dockerfile_path="pipelines/sources/lead_conversion",
            step_name="inference",
            command=["python3", "simple_step.py"],
            instance_type=instance_type_var,
            role=role,
            sagemaker_session=sm_session,
        ).create_processor()

        inference_step = ProcessingStep(
            name="InferenceStep",
            processor=inference_processor,
            inputs=inputs,
            outputs=outputs,
            code=script_path,
        )

        # Definir los pasos del pipeline
        steps = [data_prep_step, inference_step]

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
