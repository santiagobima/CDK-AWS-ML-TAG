from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker,
    aws_ssm as ssm,
    aws_iam as iam
)
from constructs import Construct
import os



class ServerlessEndpointStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        model_stage: str,
        env,
        sm_execution_role_arn: str,
        pipeline_name: str,
        **kwargs
    ):
        super().__init__(scope, id, env=env, **kwargs)

        prefix = self.node.try_get_context("resource_prefix")
        bucket_name = os.getenv("DATA_BUCKET")
        if not bucket_name:
            raise ValueError("❌ DATA_BUCKET environment variable is not set.")

        image_uri = ssm.StringParameter.value_for_string_parameter(
            self, f"/{prefix}/ProcessorImageUri"
        )

        tarball_path = f"s3://{bucket_name}/output-data/predict/models/tar_models/{model_stage}.tar.gz"

        # ✅ Usamos el pipeline_name correctamente
        model = sagemaker.CfnModel(
            self, f"{model_stage}ServerlessModel",
            execution_role_arn=sm_execution_role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image_uri,
                model_data_url=tarball_path,
                environment={
                    "SAGEMAKER_PROGRAM": "pipelines/lead_conversion_rate/steps/inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": tarball_path
                }
            ),
            model_name=f"{pipeline_name}-{model_stage}-ServerlessModel"
        )

        endpoint_config = sagemaker.CfnEndpointConfig(
            self, f"{model_stage}ServerlessConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.model_name,
                    variant_name="AllTraffic",
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                        memory_size_in_mb=4096,
                        max_concurrency=10
                    )
                )
            ],
            endpoint_config_name=f"{pipeline_name}-{model_stage}-ServerlessConfig"
        )

        sagemaker.CfnEndpoint(
            self, f"{model_stage}ServerlessEndpoint",
            endpoint_config_name=endpoint_config.endpoint_config_name,
            endpoint_name=f"{pipeline_name}-{model_stage}-Endpoint"
        )