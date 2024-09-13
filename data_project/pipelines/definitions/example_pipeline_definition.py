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
        # Define a parameter for configuring the instance type
        instance_type_var = parameters.ParameterString(
            name="InstanceType",
            default_value="local" if isinstance(sm_session, LocalSession) else "ml.m5.large"
        )

        # Use the SKLearn image provided by AWS SageMaker
        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=sm_session.boto_region_name,
            version="0.23-1",
        )

        # Create a ScriptProcessor and add code / run parameters
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type_var,
            instance_count=1,
            role=role,
            sagemaker_session=sm_session,
        )

        processing_step = ProcessingStep(
            name="processing-example",
            step_args=processor.run(
                code="pipelines/sources/example_pipeline/evaluate.py",

            ),
            
            job_arguments=[
                "--config-parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )
        
        processing_step_2 = ProcessingStep(
            name="inference",
            
        )

        return Pipeline(
            name=pipeline_name,
            steps=[processing_step, processing_step_2],
            sagemaker_session=sm_session,
            parameters=[instance_type_var],
        )




"""The ExamplePipeline class implements a specific SageMaker pipeline, inheriting from SagemakerPipelineFactory.
It defines an instance type parameter that can be configured at runtime, depending on whether the session is local or cloud-based.
The pipeline uses the scikit-learn image provided by AWS to run a Python script (evaluate.py) in a ScriptProcessor.
A processing step is created using the ScriptProcessor, and the custom configuration parameter (pipeline_config_parameter) is passed as an argument to the script."""