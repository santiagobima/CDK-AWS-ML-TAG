import os
import logging
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.session import Session
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from Constructors.pipeline_factory import SagemakerPipelineFactory, get_processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LeadConversionFactory(SagemakerPipelineFactory):
    local_mode: bool = False

    class Config:
        arbitrary_types_allowed = True

    def create(self, scope, role: str, pipeline_name: str, sm_session: Session, update: bool = False) -> Pipeline:
        instance_type_var = ParameterString(
            name="InstanceType",
            default_value="local" if self.local_mode else "ml.m5.4xlarge"
        )
        logger.info(f"Modo local: {self.local_mode}")

        data_bucket_name = os.getenv("DATA_BUCKET").rstrip('/')
        inputs, _ = self._configure_io(data_bucket_name)

        processor = get_processor(
            role=role,
            instance_type=instance_type_var.default_value
        )

        data_prep_step = ProcessingStep(
            name='Temporary_Simple_Check_Step',
            processor=processor,
            inputs=inputs,
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{data_bucket_name}/output-data/simple_step"
                )
            ],
            code="pipelines/lead_conversion_rate/steps/simple_step.py",
            cache_config=CacheConfig(enable_caching=True, expire_after="7d")
        )

        retrieve_data_step = ProcessingStep(
            name='RetrieveDataStep',
            processor=processor,
            inputs=inputs,
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/retrieve",
                    destination=f"s3://{data_bucket_name}/output-data/retrieve"
                )
            ],
            code="pipelines/lead_conversion_rate/steps/data_read.py",
            job_arguments=["--environment", os.getenv('ENV', 'dev')],
            cache_config=CacheConfig(enable_caching=True, expire_after="7d")
        )

        prep_data_step = ProcessingStep(
            name='PreprocessDataStep',
            processor=processor,
            inputs=inputs + [
                ProcessingInput(
                    source=f"s3://{data_bucket_name}/output-data/retrieve",
                    destination="/opt/ml/processing/retrieve"
                )
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{data_bucket_name}/output-data"
                )
            ],
            code="pipelines/lead_conversion_rate/steps/data_prep.py",
            job_arguments=[
                "--input_path", "/opt/ml/processing/retrieve/train.pkl",
                "--output_path", "/opt/ml/processing/output/baseline_features_raw.pkl"
            ],
            cache_config=CacheConfig(enable_caching=True, expire_after="7d")
        )
        
        predict_step = ProcessingStep(
            name='PredictStep',
            processor=processor,
            inputs=inputs + [
                ProcessingInput(
                    source=f"s3://{data_bucket_name}/output-data",
                    destination="/opt/ml/processing/predict_input_data"
                ),
                ProcessingInput(
                    source=f"s3://{data_bucket_name}/configs/model_config.yml",
                    destination="/opt/ml/processing/configs"
                ),
                
                ProcessingInput(
                    source=f"s3://{data_bucket_name}/code/source_code/pipelines/lead_conversion_rate/model/summaries",
                    destination="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/summaries"
                )

                
            ],
            outputs=[
                ProcessingOutput(  # modelos entrenados
                    source="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/pickles/models",
                    destination=f"s3://{data_bucket_name}/output-data/predict/models"
                ),
                ProcessingOutput(  # features seleccionadas
                    source="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/pickles/features",
                    destination=f"s3://{data_bucket_name}/output-data/predict/features"
                ),
                ProcessingOutput(  # métricas y resultados
                    source="/opt/ml/processing/source_code/pipelines/lead_conversion_rate/model/results",
                    destination=f"s3://{data_bucket_name}/output-data/predict/results"
                )
            ],
            code="pipelines/lead_conversion_rate/steps/predict.py",
            cache_config=CacheConfig(enable_caching=True, expire_after="7d")
        )
            
        retrieve_data_step.add_depends_on([data_prep_step])
        prep_data_step.add_depends_on([retrieve_data_step])
        predict_step.add_depends_on([prep_data_step])
        steps = [data_prep_step, retrieve_data_step, prep_data_step, predict_step]

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
            ),
            ProcessingInput(
                source=f"s3://{data_bucket_name}/code/source_code/summaries",
                destination="/opt/ml/processing/summaries"
            )
        ]
        return inputs, []