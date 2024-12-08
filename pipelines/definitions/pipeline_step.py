from constructs import Construct
from aws_cdk.aws_ecr_assets import DockerImageAsset
from sagemaker import ScriptProcessor

class PipelineStep:
    def __init__(self, scope: Construct, id: str, dockerfile_path: str, step_name: str, command: list, instance_type: str, role: str, sagemaker_session):
        """
        Clase para gestionar un paso del pipeline con imágenes Docker personalizadas.

        :param scope: Alcance del constructo (para la integración con CDK).
        :param id: Identificador único del paso.
        :param dockerfile_path: Ruta al Dockerfile.
        :param step_name: Nombre del paso.
        :param command: Comando a ejecutar en el contenedor.
        :param instance_type: Tipo de instancia para SageMaker.
        :param role: ARN del rol de ejecución de SageMaker.
        :param sagemaker_session: Sesión de SageMaker.
        """
        self.scope = scope
        self.id = id
        self.dockerfile_path = dockerfile_path
        self.step_name = step_name
        self.command = command
        self.instance_type = instance_type
        self.role = role
        self.sagemaker_session = sagemaker_session

    def create_sagemaker_image(self):
        """
        Crea la imagen Docker usando CDK y devuelve su URI.

        :return: URI de la imagen Docker creada.
        """
        asset = DockerImageAsset(
            scope=self.scope,
            id=self.id,
            directory=self.dockerfile_path
        )
        return asset.image_uri

    def create_processor(self):
        """
        Configura el procesador de SageMaker con la imagen creada.

        :return: Instancia de ScriptProcessor configurada.
        """
        image_uri = self.create_sagemaker_image()
        return ScriptProcessor(
            image_uri=image_uri,
            command=self.command,
            instance_type=self.instance_type,
            instance_count=1,
            role=self.role,
            sagemaker_session=self.sagemaker_session
        )
