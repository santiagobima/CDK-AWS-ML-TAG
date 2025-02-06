import logging
import sagemaker
from sagemaker.workflow.pipeline import Pipeline

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def execute_cloud_pipeline():
    """
    Ejecuta el pipeline en la nube utilizando SageMaker.
    """
    session = sagemaker.Session()
    pipeline_name = "example-pipeline"
    pipeline = Pipeline(name=pipeline_name, sagemaker_session=session)

    logger.info(f"Ejecutando pipeline '{pipeline_name}' en la nube...")
    execution = pipeline.start()

    # Esperar a que el pipeline termine
    execution.wait()
    logger.info("Pipeline execution completed in the cloud.")

if __name__ == "__main__":
    execute_cloud_pipeline()
