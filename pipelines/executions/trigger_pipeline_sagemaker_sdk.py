import os
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession
import sys

# Añadir el directorio raíz del proyecto al PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
print(f"Directorio raíz añadido al PYTHONPATH: {root_dir}")  # Verificar si es correcto

from pipelines.definitions.lead_conversion_pipeline import LeadConversionFactory  


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

if local_mode:
    # Aquí, en lugar de ejecutar pipeline.start(), ejecutamos manualmente los pasos del pipeline.
    print(f"Ejecutando pasos de pipeline {pipeline_name} localmente...")

    # Aquí es donde deberías definir los pasos del pipeline
    # Por ejemplo, podrías invocar directamente los scripts que definen tus pasos
    os.system("python3 pipelines/sources/lead_conversion/evaluate.py")
    os.system("python3 pipelines/sources/lead_conversion/simple_step.py")

    print("Ejecución local del pipeline completada.")
else:
    # Para la nube
    pipeline = Pipeline(name=pipeline_name, sagemaker_session=session)
    execution = pipeline.start()

    # Esperar a que el pipeline termine
    execution.wait()
    print("Pipeline execution completed.")
