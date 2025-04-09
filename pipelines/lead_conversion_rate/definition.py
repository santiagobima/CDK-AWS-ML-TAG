import os
import logging
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.session import Session
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from Constructors.pipeline_factory import SagemakerPipelineFactory, get_processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LeadConversionFactory(SagemakerPipelineFactory):
    """
    Clase que define el pipeline de conversión de clientes potenciales.
    """
    local_mode: bool = False

    class Config:
        arbitrary_types_allowed = True

    def create(self, scope, role: str, pipeline_name: str, sm_session: Session, image_uri: str, update: bool = False) -> Pipeline:
        """
        Crea el pipeline de SageMaker.
        """
        instance_type_var = ParameterString(
            name="InstanceType",
            default_value="local" if self.local_mode else "ml.m5.large"
        )
        logger.info(f"Modo local: {self.local_mode}")

        data_bucket_name = os.getenv("DATA_BUCKET").rstrip('/')
        inputs, outputs = self._configure_io(data_bucket_name)
        
        processor = get_processor(role = role, instance_type = instance_type_var.default_value, image_uri=image_uri)
        
        data_prep_step = ProcessingStep(
            name='DataPreparationStep',
            processor=processor,
            inputs=inputs,
            outputs=outputs,
            code="pipelines/lead_conversion_rate/steps/simple_step.py"
            
            
        )
        
        
        retrieve_data_step=ProcessingStep(
            name='RetrieveDataStep',
            processor=processor,
            inputs=[],  # 
            outputs=outputs,  # 
            code="pipelines/lead_conversion_rate/steps/data_read.py",
        )
        
        retrieve_data_step.add_depends_on([data_prep_step])
        steps = [data_prep_step,retrieve_data_step]
        logger.info(f"Pipeline '{pipeline_name}' configurado con {len(steps)} paso(s).")
        return Pipeline(name=pipeline_name, steps=steps, sagemaker_session=sm_session)
        

    """    # Paso de preparación de datos con SKLearnProcessor (sin Docker)
        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=role,
            instance_type=instance_type_var.default_value,  # 🔹 Corregido: debe ser un string, no ParameterString
            instance_count=1,
            sagemaker_session=sm_session
        )

        data_prep_step = ProcessingStep(
            name="DataPreparationStep",
            processor=sklearn_processor,
            inputs=inputs,
            outputs=outputs,
            code="pipelines/lead_conversion_rate/sources/simple_step.py"
        )

        # Paso de consulta a Athena (sin Docker)
        athena_script_path = "pipelines/lead_conversion_rate/sources/athena_query.py"
        athena_processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=role,
            instance_type=instance_type_var.default_value,  # 🔹 Corregido
            instance_count=1,
            sagemaker_session=sm_session
        )

        retrieve_data_step = ProcessingStep(
            name="RetrieveDataStep",
            processor=athena_processor,
            inputs=[],
            outputs=outputs,
            code=athena_script_path
        )

        retrieve_data_step.add_depends_on([data_prep_step])
        steps = [data_prep_step, retrieve_data_step]

        logger.info(f"Pipeline '{pipeline_name}' configurado con {len(steps)} paso(s).")

        return Pipeline(
            name=pipeline_name,
            steps=steps,
            sagemaker_session=sm_session,
            parameters=[instance_type_var],
        )
"""

    def _configure_io(self, data_bucket_name: str):
        """
        Configura inputs y outputs en S3.
        """
        logger.info(f"Configurando pipeline con bucket S3 '{data_bucket_name}' para entrada y salida.")
        inputs = [
            ProcessingInput(
                source=f"s3://{data_bucket_name}/input-data",
                destination="/opt/ml/processing/input"
            )
        ]
        outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/output/*",  #---->> I have just change it with Edo.
                destination=f"s3://{data_bucket_name}/output-data"
            )
        ]
        return inputs, outputs