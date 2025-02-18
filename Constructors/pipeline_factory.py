import os
import logging
import boto3
import sagemaker

from constructs import Construct
from abc import ABC, abstractmethod
from sagemaker.workflow.pipeline import Pipeline  # Importación directa de Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SagemakerPipelineFactory(BaseModel):
    local_mode: bool = False

    class Config:
        arbitrary_types_allowed = True  # ✅ Permite clases abstractas en Pydantic

    @abstractmethod
    def create(
        self,
        scope: Construct,
        role: str,
        pipeline_name: str,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        raise NotImplementedError("Debe implementar el método 'create' en la subclase.")

def create_sagemaker_session(default_bucket: str, local_mode=False) -> sagemaker.Session:
    """
    Crea una sesión de SageMaker, en local si `local_mode=True`.
    """
    region = os.getenv('CDK_DEFAULT_REGION')
    boto_session = boto3.Session(region_name=region)

    try:
        if local_mode:
            sagemaker_session = LocalPipelineSession(default_bucket=default_bucket)
            logger.info("Modo local activado para SageMaker.")
        else:
            sagemaker_client = boto_session.client("sagemaker")
            sagemaker_session = PipelineSession(
                boto_session=boto_session,
                sagemaker_client=sagemaker_client,
                default_bucket=default_bucket,
            )
            logger.info("Sesión de SageMaker en la nube activada.")
    except Exception as e:
        logger.exception("Error al crear la sesión de SageMaker")
        raise RuntimeError(f"No se pudo crear la sesión de SageMaker: {e}")

    return sagemaker_session
