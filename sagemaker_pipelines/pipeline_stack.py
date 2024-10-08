import os
import json
from typing import Tuple

import aws_cdk as cdk
import sagemaker
from aws_cdk import (
    aws_sagemaker as sm,
    aws_ssm as ssm,
)
from constructs import Construct

from pipelines.definitions.base import SagemakerPipelineFactory, create_sagemaker_session

class PipelineStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        factory: SagemakerPipelineFactory,
        **kwargs
    ) -> None:
        # Definir explícitamente account y region
        env_account = scope.node.try_get_context("account")
        env_region = scope.node.try_get_context("region")

        # Si estamos en modo local, usar la región del bucket S3 (eu-west-1)
        local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'
        if local_mode:
            env_region = "eu-west-1"  # Definir la región del bucket S3 en modo local

        super().__init__(scope, id, env={
            "account": env_account,
            "region": env_region,
        }, **kwargs)

        self.factory = factory
        self.prefix = self.node.try_get_context("resource_prefix")

        # Mostrar que estamos en modo local
        if local_mode:
            print("Ejecutando en modo local")

        # Cargar nombres de recursos desde SSM Parameter Store (solo si no estamos en modo local)
        if not local_mode:
            sources_bucket_name = ssm.StringParameter.value_from_lookup(
                self, f"/{self.prefix}/SourcesBucketName")
            sm_execution_role_arn = ssm.StringParameter.value_from_lookup(
                self, f"/{self.prefix}/SagemakerExecutionRoleArn")
        else:
            # Utilizar el bucket de S3 en eu-west-1 para almacenar resultados en modo local
            sources_bucket_name = "awsbucketsb"  # Tu bucket de S3
            sm_execution_role_arn = "arn:aws:iam::123456789012:role/local-role"  # Cambia el ARN si es necesario

        # Crear el pipeline configurado
        self.lead_conversion, self.lead_conversion_arn = self.create_pipeline(
            pipeline_name='example-pipeline',
            pipeline_factory=self.factory,
            sources_bucket_name=sources_bucket_name,
            sm_execution_role_arn=sm_execution_role_arn,
        )

    def create_pipeline(
        self,
        pipeline_name: str,
        pipeline_factory: SagemakerPipelineFactory,
        sources_bucket_name: str,
        sm_execution_role_arn: str,
    ) -> Tuple[sm.CfnPipeline, str]:
        # Determinar si ejecutar en modo local
        local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

        # Crear la sesión de SageMaker (local o en la nube)
        sm_session: sagemaker.Session = create_sagemaker_session(
            region=self.region,  # Se asegura que la región esté bien definida
            default_bucket=sources_bucket_name,  # Ahora siempre usa tu bucket "awsbucketsb"
            local_mode=local_mode
        )

        # Definir el pipeline
        if 'dummy-value-for-' in sources_bucket_name:
            pipeline_def_json = '{}'
        else:
            pipeline = pipeline_factory.create(
                pipeline_name=pipeline_name,
                role=sm_execution_role_arn,  # Usar el mismo rol para ambos modos
                sm_session=sm_session,
            )

            pipeline_def_json = json.dumps(json.loads(pipeline.definition()), indent=2, sort_keys=True)

        # Definir el recurso CloudFormation para el pipeline
        pipeline_cfn = sm.CfnPipeline(
            self,
            id=f"SagemakerPipeline-{pipeline_name}",
            pipeline_name=pipeline_name,
            pipeline_definition={"PipelineDefinitionBody": pipeline_def_json},
            role_arn=sm_execution_role_arn,
        )
        arn = self.format_arn(
            service='sagemaker',
            resource='pipeline',
            resource_name=pipeline_cfn.pipeline_name,
        )
        return pipeline_cfn, arn
