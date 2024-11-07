# pipeline_stack.py
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
        env: cdk.Environment,
        local_mode: bool = False,
        **kwargs
    ) -> None:
        super().__init__(scope, id, env=env, **kwargs)

        self.factory = factory
        self.prefix = self.node.try_get_context("resource_prefix")
        local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'
        #self.local_mode = local_mode --> es el que tengo que activar y comentar el de arriba
        env_region = "eu-west-1" if local_mode else self.region

        # Cargar nombres de recursos desde SSM Parameter Store (solo si no estamos en modo local)
        if not local_mode:
            try:
                sources_bucket_name = ssm.StringParameter.value_from_lookup(
                    self, f"/{self.prefix}/SourcesBucketName")
                sm_execution_role_arn = ssm.StringParameter.value_from_lookup(
                    self, f"/{self.prefix}/SagemakerExecutionRoleArn")
            except Exception as e:
                print(f"Error al obtener parámetros SSM: {e}")
                raise ValueError("Parámetros SSM no disponibles. Asegúrate de que `DSM-SagemakerStack` se haya desplegado primero.")
        else:
            sources_bucket_name = "awsbucketsb"
            sm_execution_role_arn = "arn:aws:iam::123456789012:role/local-role"

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
        local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'
        sm_session: sagemaker.Session = create_sagemaker_session(
            region=self.region,
            default_bucket=sources_bucket_name,
            local_mode=local_mode
        )

        if 'dummy-value-for-' in sources_bucket_name:
            pipeline_def_json = '{}'
        else:
            pipeline = pipeline_factory.create(
                pipeline_name=pipeline_name,
                role=sm_execution_role_arn,
                sm_session=sm_session,
            )
            pipeline_def_json = json.dumps(json.loads(pipeline.definition()), indent=2, sort_keys=True)

        if local_mode:
            print("Ejecutando en modo local. No se creará el recurso SageMaker::Pipeline.")
            arn = f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{pipeline_name}"
            return None, arn

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
