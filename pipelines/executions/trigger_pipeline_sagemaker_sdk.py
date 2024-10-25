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

#REMOVE IF LOCAL AND ELSE. WE WILL DO IT IN LOCAL MODE ALL THE TIME. IF the pipeline does not exist in aws we cant run. the in put and ouput will be in awws. the only thing is that the power will be used in local.  The line 10 we can remove also. If we run this script it's a local mode.

if local_mode:
    # Aquí, en lugar de ejecutar pipeline.start(), ejecutamos manualmente los pasos del pipeline.
    print(f"Ejecutando pasos de pipeline {pipeline_name} localmente...")

    # Ejecución del script evaluate.py pasando argumentos
    os.system("python3 pipelines/sources/lead_conversion/evaluate.py --config_parameter 'Cloud Developer' --name 'Santiago'")

    # Ejecución del siguiente paso
    os.system("python3 pipelines/sources/lead_conversion/simple_step.py")

    print("Ejecución local del pipeline completada.")
else:
    # Para la nube
    pipeline = Pipeline(name=pipeline_name, sagemaker_session=session)
    execution = pipeline.start()

    # Esperar a que el pipeline termine
    execution.wait()
    print("Pipeline execution completed.")
