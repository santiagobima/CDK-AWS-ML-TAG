# run_pipeline_locally.py

import boto3
import sagemaker
from sagemaker.local import LocalSession
from pipelines.definitions.lead_conversion_pipeline import LeadConversionFactory

session = #check how boto3 how to return a session and then use it in line 12.

def main():
    # Crear una sesión local de SageMaker
    sm_session = LocalSession(boto_session=)
    sm_session.config = {'local': {'local_code': True}}

    # Definir el rol de IAM (puedes usar un rol ficticio en modo local)
    role = "arn:aws:iam::123456789012:role/SageMakerRole"

    # Crear una instancia de tu fábrica de pipelines con el parámetro requerido
    factory = LeadConversionFactory(pipeline_config_parameter="tu_valor_aquí")

    # Crear el pipeline
    pipeline = factory.create(
        role=role,
        pipeline_name="lead-conversion-pipeline",
        sm_session=sm_session,
    )

    # Ejecutar el pipeline
    execution = pipeline.run()

    # Verificar si estamos en modo local
    if sm_session.local_mode:
        # En modo local, la ejecución es sincrónica y no es necesario llamar a `wait()`
        print("Pipeline ejecutado en modo local.")
    else:
        # En modo nube, podemos llamar a `wait()` para esperar a que termine
        execution.wait()
        print("Pipeline ejecutado en modo nube.")

if __name__ == "__main__":
    main()

