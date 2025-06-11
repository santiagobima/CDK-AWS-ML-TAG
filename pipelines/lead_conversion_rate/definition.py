import os
import logging
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.session import Session
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
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

    def create(self, scope, role: str, pipeline_name: str, sm_session: Session, update: bool = False) -> Pipeline:
        """
        Crea el pipeline de SageMaker.
        """
        instance_type_var = ParameterString(
            name="InstanceType",
            default_value="local" if self.local_mode else "ml.m5.4xlarge"
        )
        logger.info(f"Modo local: {self.local_mode}")

        data_bucket_name = os.getenv("DATA_BUCKET").rstrip('/')
        inputs, outputs = self._configure_io(data_bucket_name)

        # Usar imagen gestionada por AWS (SKLearnProcessor)
        processor = get_processor(
            role=role,
            instance_type=instance_type_var.default_value
        )

        data_prep_step = ProcessingStep(
            name='Temporary_Simple_Check_Step',
            processor=processor,
            inputs=inputs,
            outputs=outputs,
            code="pipelines/lead_conversion_rate/steps/simple_step.py"
        )

        retrieve_data_step = ProcessingStep(
            name='RetrieveDataStep',
            processor=processor,
            inputs=inputs,
            outputs=outputs,
            code="pipelines/lead_conversion_rate/steps/data_read.py",  #here I did a change
            job_arguments=["--environment", os.getenv('ENV', 'dev')], 
        ) 
        
        retrieve_data_step.add_depends_on([data_prep_step])
        steps = [data_prep_step, retrieve_data_step]

        logger.info(f"Pipeline '{pipeline_name}' configurado con {len(steps)} paso(s).")
        return Pipeline(name=pipeline_name, steps=steps, sagemaker_session=sm_session)

    def _configure_io(self, data_bucket_name: str):
            logger.info(f"Configurando inputs/outputs con bucket {data_bucket_name}")
            inputs = [
                ProcessingInput(
                    source=f"s3://{data_bucket_name}/input-data",
                    destination="/opt/ml/processing/input"
                ),
                
                ProcessingInput(
                    source=f"s3://{data_bucket_name}/code/source_code",
                    destination="/opt/ml/processing/source_code"
                    )
              
            ]
            outputs = [
                ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{data_bucket_name}/output-data"
                )
            ]
            return inputs, outputs