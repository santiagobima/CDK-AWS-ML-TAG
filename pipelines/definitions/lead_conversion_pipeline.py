import sagemaker
import sagemaker.image_uris
from sagemaker import LocalSession, ScriptProcessor
from sagemaker.workflow import parameters
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from pipelines.definitions.base import SagemakerPipelineFactory
import os

class LeadConversionFactory(SagemakerPipelineFactory):
    pipeline_config_parameter: str

    def create(
        self,
        role: str,
        pipeline_name: str,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        # Definir una variable de entorno para ejecutar en local o no
        local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

        # Definir un parámetro para configurar el tipo de instancia
        instance_type_var = parameters.ParameterString(
            name="InstanceType",
            default_value="local" if isinstance(sm_session, LocalSession) else "ml.m5.large"
        )

        # Usar la imagen de SKLearn proporcionada por AWS SageMaker
        image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=sm_session.boto_region_name,
            version="0.23-1",
        )

        # Crear un ScriptProcessor y agregar código/parámetros de ejecución
        processor = ScriptProcessor(
            image_uri=image_uri,
            command=["python3"],
            instance_type=instance_type_var,
            instance_count=1,
            role=role,
            sagemaker_session=sm_session,
        )

        # Paso 1: Ejemplo de paso de procesamiento
        if local_mode:
            code_path = os.path.abspath("./pipelines/sources/lead_conversion/evaluate.py")
            inputs = []  # No inputs for local mode
            outputs = []  # No outputs for local mode
        else:
            # En modo no local, usa S3
            code_path = "pipelines/sources/lead_conversion/evaluate.py"
            
            # Definir inputs y outputs para procesamiento en la nube
            inputs = [
                sagemaker.processing.ProcessingInput(
                    source='s3://dsa-sm-data/input-data',  # Ruta de los datos en S3
                    destination='/opt/ml/processing/input'  # Directorio en el contenedor
                )
            ]
            outputs = [
                sagemaker.processing.ProcessingOutput(
                    source='/opt/ml/processing/output',  # Directorio donde se generan los resultados
                    destination='s3://dsa-sm-data/output-data'  # Ruta en S3 donde guardar los resultados
                )
            ]

        processing_step = ProcessingStep(
            name="processing-example",
            step_args=processor.run(
                code=code_path,
                inputs=inputs,  # Pasa los inputs aquí
                outputs=outputs  # Pasa los outputs aquí
            ),
            job_arguments=[
                "--config-parameter", self.pipeline_config_parameter,
                "--name", "santiago"
            ],
        )

        # Paso 2: Paso de procesamiento local sin S3
        processing_step_2 = ProcessingStep(
            name="local-processing-step",
            step_args=processor.run(
                code="pipelines/sources/lead_conversion/simple_step.py",
                inputs=[],  # Sin entradas
                outputs=[],  # Sin salidas
            ),
        )

        # Definir los pasos a incluir en el pipeline según el modo (local o no local)
        if local_mode:
            steps = [processing_step_2]  # Solo el paso local
        else:
            steps = [processing_step]  # Solo el paso no local

        # Definir el pipeline con los pasos apropiados
        return Pipeline(
            name=pipeline_name,
            steps=steps,  # Incluir solo los pasos relevantes según el modo
            sagemaker_session=sm_session,
            parameters=[instance_type_var],
        )


"""This error is thrown because in SageMaker's ProcessingStep, either step_args or processor is required, but not both at the same time.In your lead_conversion_definition.py, you're defining processing_step_2 without the required arguments:"""
"""The ExamplePipeline class implements a specific SageMaker pipeline, inheriting from SagemakerPipelineFactory.
""It defines an instance type parameter that can be configured at runtime, depending on whether the session is local or cloud-based.
The pipeline uses the scikit-learn image provided by AWS to run a Python script (evaluate.py) in a ScriptProcessor.
A processing step is created using the ScriptProcessor, and the custom configuration parameter (pipeline_config_parameter) is passed as an argument to the script."""