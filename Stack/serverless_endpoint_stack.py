from aws_cdk import (
    Stack,
    aws_sagemaker as sagemaker,
)
from constructs import Construct
import os
import boto3


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

        region = os.getenv("CDK_DEFAULT_REGION")
        if not region:
            raise ValueError("❌ Falta la variable CDK_DEFAULT_REGION")

        model_package_group_name = f"{pipeline_name}-Group"
        MAX_NAME_LENGTH = 63
        base_model_name = f"{pipeline_name}-{model_stage}-Model"
        model_name = base_model_name[:MAX_NAME_LENGTH]
        

        # 🔍 Obtener el último modelo aprobado
        sm_client = boto3.client("sagemaker", region_name=region)
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy="CreationTime",
            SortOrder="Descending"
        )
        approved_models = [
            pkg for pkg in response["ModelPackageSummaryList"]
            if pkg["ModelApprovalStatus"] == "Approved"
        ]
        if not approved_models:
            raise ValueError(f"No hay modelos aprobados en el grupo '{model_package_group_name}'.")

        model_package_arn = approved_models[0]["ModelPackageArn"]

        # ✅ Crear CfnModel con model_package_name
        model = sagemaker.CfnModel(
            self, f"{model_stage}ServerlessModel",
            execution_role_arn=sm_execution_role_arn,
            containers=[
                sagemaker.CfnModel.ContainerDefinitionProperty(
                    model_package_name=model_package_arn
                )
            ],
            model_name=model_name
        )

        # ✅ Crear EndpointConfig que depende de ese modelo
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, f"{model_stage}ServerlessConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=model.model_name,  # <-- ⚠️ IMPORTANTE: usar .ref para obtener el logical ID
                    variant_name="AllTraffic",
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                        memory_size_in_mb=4096,
                        max_concurrency=10
                    )
                )
            ],
            endpoint_config_name=f"{pipeline_name}-{model_stage}-ServerlessConfig"
        )
        endpoint_config.add_dependency(model)  # 🔒 asegúrate de que se cree luego del modelo

        # ✅ Crear endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, f"{model_stage}ServerlessEndpoint",
            endpoint_config_name=endpoint_config.endpoint_config_name,
            endpoint_name=f"{pipeline_name}-{model_stage}-Endpoint"
        )
        endpoint.add_dependency(endpoint_config)