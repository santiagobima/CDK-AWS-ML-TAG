import os
import logging
import aws_cdk as cdk
from aws_cdk import aws_iam as iam, aws_s3 as s3, aws_ec2 as ec2, aws_ssm as ssm
from constructs import Construct

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SagemakerStack(cdk.Stack):
    """
    Stack de CDK para los recursos de SageMaker, incluyendo roles y buckets necesarios para los pipelines.
    """
    def __init__(
        self,
        scope: Construct,
        id: str,
        env: cdk.Environment,
        vpc_name: str,
        local_mode: bool,  # Recibe `local_mode` desde app.py
        **kwargs
    ) -> None:
        super().__init__(scope, id, env=env, **kwargs)
        
        self.prefix = self.node.try_get_context("resource_prefix")

        # Lookup de la VPC
        self.vpc = ec2.Vpc.from_lookup(self, id=f"{self.prefix}-VpcLookup", vpc_id=vpc_name)
        logger.info(f"VPC '{vpc_name}' cargada exitosamente.")

        # Crear el rol de ejecución de SageMaker
        self.sm_execution_role = self.create_execution_role()

        # Obtener o crear buckets de datos y fuentes
        self.sm_data_bucket = self.get_or_create_bucket(
            bucket_name=os.getenv("DATA_BUCKET"),
            bucket_id="DataBucket",
            description="Data bucket name for SageMaker"
        )

        self.sm_sources_bucket = self.get_or_create_bucket(
            bucket_name=os.getenv("SOURCES_BUCKET"),
            bucket_id="SourcesBucket",
            description="Sources bucket name for SageMaker"
        )

        # Conceder acceso de lectura/escritura en los buckets
        self.sm_data_bucket.grant_read_write(self.sm_execution_role)
        self.sm_sources_bucket.grant_read(self.sm_execution_role)

        # Crear parámetros en SSM
        self.create_ssm_parameters()

    def create_execution_role(self) -> iam.Role:
        """
        Crea el rol de ejecución de SageMaker con los permisos necesarios.

        :return: El rol de IAM creado.
        """
        role = iam.Role(
            self, 'SagemakerExecutionRole',
            assumed_by=iam.ServicePrincipal('sagemaker.amazonaws.com'),
            role_name=f"{self.prefix}-sm-execution-role",
            managed_policies=[
                iam.ManagedPolicy.from_managed_policy_arn(
                    self,
                    id="SagemakerFullAccess",
                    managed_policy_arn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                ),
            ],
        )
        logger.info("Rol de ejecución de SageMaker creado con éxito.")

        # Agregar permisos específicos para los buckets
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:ListBucket", "s3:GetBucketLocation",
                "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "glue:GetTable",
                "glue:GetDatabase"
                
            ],
            resources=[
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}/*",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}/*"
            ]
        ))
        return role

    def get_or_create_bucket(self, bucket_name: str, bucket_id: str, description: str) -> s3.Bucket:
        """
        Obtiene o crea un bucket de S3 por su nombre.

        :param bucket_name: Nombre del bucket especificado en el entorno.
        :param bucket_id: Identificador del bucket.
        :param description: Descripción para el parámetro SSM.
        :return: Instancia del bucket de S3.
        """
        if bucket_name:
            try:
                logger.info(f"Intentando cargar bucket '{bucket_name}'.")
                return s3.Bucket.from_bucket_name(self, bucket_id, bucket_name=bucket_name)
            except Exception as e:
                logger.warning(f"Bucket '{bucket_name}' no encontrado. Error: {e}")

        # Crear nuevo bucket si no existe
        logger.info(f"Creando nuevo bucket con ID '{bucket_id}' ya que '{bucket_name}' no se encontró.")
        return s3.Bucket(
            self,
            id=bucket_id,
            bucket_name=f"{self.prefix}-{bucket_id.lower()}-{self.account}",
            versioned=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            access_control=s3.BucketAccessControl.PRIVATE,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            public_read_access=False,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

    def create_ssm_parameters(self):
        """
        Crea los parámetros necesarios en SSM Parameter Store para los nombres de los buckets
        y el ARN del rol de ejecución.
        """
        ssm.StringParameter(
            self, 'DataBucketName',
            parameter_name=f"/{self.prefix}/DataBucketName",
            string_value=self.sm_data_bucket.bucket_name,
            description="Data bucket name for SageMaker"
        )
        logger.info(f"Parámetro SSM '/{self.prefix}/DataBucketName' creado.")

        ssm.StringParameter(
            self, 'SourcesBucketName',
            parameter_name=f"/{self.prefix}/SourcesBucketName",
            string_value=self.sm_sources_bucket.bucket_name,
            description="Sources bucket name for SageMaker"
        )
        logger.info(f"Parámetro SSM '/{self.prefix}/SourcesBucketName' creado.")

        ssm.StringParameter(
            self, 'SagemakerExecutionRoleArn',
            parameter_name=f"/{self.prefix}/SagemakerExecutionRoleArn",
            string_value=self.sm_execution_role.role_arn,
            description="SageMaker Execution Role ARN"
        )
        logger.info(f"Parámetro SSM '/{self.prefix}/SagemakerExecutionRoleArn' creado.")
