import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from botocore.exceptions import NoCredentialsError, ClientError

# Verificar si estamos en modo local
local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

try:
    # Crear la sesión de SageMaker
    if local_mode:
        print("Ejecutando el pipeline en modo local")
        session = LocalPipelineSession()
    else:
        print("Ejecutando el pipeline en la nube")
        session = sagemaker.Session()

    # Definir el nombre del pipeline
    pipeline_name = "example-pipeline"

    # Cargar el pipeline
    pipeline = Pipeline(name=pipeline_name, sagemaker_session=session)

    # Iniciar la ejecución del pipeline
    print(f"Iniciando la ejecución del pipeline: {pipeline_name}")
    execution = pipeline.start()

    # Esperar a que el pipeline termine
    execution.wait()
    print("Ejecución del pipeline completada con éxito")

except NoCredentialsError as e:
    print("Error: No se encontraron credenciales de AWS. Verifica tu configuración.")
except ClientError as e:
    print(f"Error al ejecutar el pipeline: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")

