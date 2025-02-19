from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.xgboost import XGBoostProcessor
from sagemaker import Session
from sagemaker.workflow.parameters import ParameterString
import logging
import os

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PipelineStep:
    def __init__(self, step_name: str, script_path: str, instance_type: str, role: str, sagemaker_session: Session, framework: str):
        """
        Clase para gestionar un paso del pipeline sin necesidad de imágenes Docker personalizadas.
        """
        self.step_name = step_name
        self.script_path = script_path
        self.instance_type = instance_type
        self.role = role
        self.sagemaker_session = sagemaker_session
        self.framework = framework.lower()

    def create_processor(self):
        """
        Crea el procesador de SageMaker adecuado según el framework especificado.
        """
        if self.framework == "sklearn":
            processor = SKLearnProcessor(
                framework_version="1.2-1",
                role=self.role,
                instance_type=self.instance_type,
                instance_count=1,
                sagemaker_session=self.sagemaker_session,
            )
            logger.info(f"Usando SKLearnProcessor para el paso '{self.step_name}'.")

        elif self.framework == "xgboost":
            processor = XGBoostProcessor(
                framework_version="1.5-1",
                role=self.role,
                instance_type=self.instance_type,
                instance_count=1,
                sagemaker_session=self.sagemaker_session,
            )
            logger.info(f"Usando XGBoostProcessor para el paso '{self.step_name}'.")

        else:
            region = os.getenv("CDK_DEFAULT_REGION", "us-east-1")  # 🔹 Ahora obtiene la región correcta
            processor = ScriptProcessor(
                image_uri=f"{region}.dkr.ecr.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                role=self.role,
                instance_type=self.instance_type,
                instance_count=1,
                command=["python3"],
                sagemaker_session=self.sagemaker_session,
            )
            logger.info(f"Usando ScriptProcessor con imagen preexistente para el paso '{self.step_name}'.")

        return processor