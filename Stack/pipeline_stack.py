import json
import logging
import aws_cdk as cdk

from typing import Tuple
from aws_cdk import aws_sagemaker
from constructs import Construct
from Constructors.pipeline_factory import SagemakerPipelineFactory, create_sagemaker_session

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PipelineStack(cdk.Stack):
    """
    Stack de CDK para la configuración de pipelines de SageMaker, 
    soporta ejecución en modo local o en la nube.
    """
    def __init__(
        self,
        scope: Construct,
        id: str,
        factory: SagemakerPipelineFactory,
        env: cdk.Environment,
        local_mode: bool,  # Recibe `local_mode` desde app.py
        pipeline_name: str,
        source_bucket_name: str,
        sm_execution_role_arn: str,
        image_uri: str,
        **kwargs
    ) -> None:
        super().__init__(scope, id, env=env, **kwargs)
        
        self.factory = factory
        self.prefix = self.node.try_get_context("resource_prefix") 
        
        # Cargar nombres de recursos desde SSM Parameter Store
        # sources_bucket_name, sm_execution_role_arn = self._load_ssm_parameters()

        # Crear el pipeline configurado
        self.pipeline, self.pipeline_arn = self.create_pipeline(
            pipeline_name=pipeline_name,
            pipeline_factory=self.factory,
            sources_bucket_name=source_bucket_name,
            sm_execution_role_arn=sm_execution_role_arn,
            local_mode=local_mode,
            image_uri=image_uri 
        )

    # # for now it works, but we'll see
    # def _load_ssm_parameters(self) -> Tuple[str, str]:
    #     """
    #     Carga los parámetros necesarios desde SSM.

    #     :return: Tupla con el nombre del bucket de fuentes y el ARN del rol de ejecución.
    #     """
    #     try:
    #         sources_bucket_name = ssm.StringParameter.value_from_lookup(
    #             self, f"/{self.prefix}/SourcesBucketName")
    #         sm_execution_role_arn = ssm.StringParameter.value_from_lookup(
    #             self, f"/{self.prefix}/SagemakerExecutionRoleArn")
    #         logger.info("Parámetros cargados exitosamente desde SSM.")
    #         return sources_bucket_name, sm_execution_role_arn
    #     except Exception as e:
    #         logger.error(f"Error al obtener parámetros SSM: {e}")
    #         raise ValueError("Parámetros SSM no disponibles. Despliega `DSM-SagemakerStack` primero.")

    def create_pipeline(
        self,
        pipeline_name: str,
        pipeline_factory: SagemakerPipelineFactory,
        sources_bucket_name: str,
        sm_execution_role_arn: str,
        local_mode: bool,
        image_uri: str
    ) -> Tuple[aws_sagemaker.CfnPipeline, str]:
        """
        Crea y configura el pipeline de SageMaker.

        :param pipeline_name: Nombre del pipeline.
        :param pipeline_factory: Fábrica de pipeline.
        :param sources_bucket_name: Nombre del bucket de fuentes.
        :param sm_execution_role_arn: ARN del rol de ejecución de SageMaker.
        :param local_mode: Si es True, se usa LocalPipelineSession.
        :return: El recurso SageMaker::Pipeline y su ARN.
        """
        # Crear sesión de SageMaker en función de local_mode
        sm_session = create_sagemaker_session(
            default_bucket=sources_bucket_name,
            local_mode=local_mode
        )

        pipeline = pipeline_factory.create(
            scope=self,
            role=sm_execution_role_arn,
            pipeline_name=pipeline_name,
            sm_session=sm_session,
            image_uri=image_uri,
            update=True,
        
        )

        pipeline_def_json = json.dumps(json.loads(pipeline.definition()), indent=2, sort_keys=True)
        logger.info(f"Definición del pipeline para '{pipeline_name}' generada con éxito.")

        # Crear el recurso CfnPipeline en la nube
        pipeline_cfn = aws_sagemaker.CfnPipeline(
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
        logger.info(f"Pipeline '{pipeline_name}' creado en la nube con ARN: {arn}.")
        return pipeline_cfn, arn
