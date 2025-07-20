import logging
import json
import sagemaker
from sagemaker.workflow.pipeline import Pipeline


# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_pipeline_name_from_cdk():
    try:
        with open("cdk.json", "r") as f:
            cdk_config = json.load(f)
            return cdk_config["context"]["pipeline_name"]
    except Exception as e:
        logger.error(f"❌ Error al leer pipeline_name desde cdk.json: {e}")
        raise



def execute_cloud_pipeline():
    """
    Ejecuta el pipeline en la nube utilizando SageMaker.
    """
    session = sagemaker.Session()
    pipeline_name = get_pipeline_name_from_cdk()
    pipeline = Pipeline(name=pipeline_name, sagemaker_session=session)

    logger.info(f"Ejecutando pipeline '{pipeline_name}' en la nube...")
    execution = pipeline.start()

    # Esperar a que el pipeline termine
    execution.wait()
    logger.info("Pipeline execution completed in the cloud.")

if __name__ == "__main__":
    execute_cloud_pipeline()
