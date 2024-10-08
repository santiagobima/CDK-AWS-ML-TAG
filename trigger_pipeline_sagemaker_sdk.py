import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession

# Verificar si estamos en modo local
local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

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
execution = pipeline.start()

# Esperar a que el pipeline termine
execution.wait()

print("Pipeline execution completed")

