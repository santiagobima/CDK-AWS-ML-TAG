import os
import aws_cdk as cdk
from aws_cdk import (
    aws_iam as iam,
    aws_s3 as s3,
    aws_ec2 as ec2,
    aws_ssm as ssm,
)
from constructs import Construct

class SagemakerStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        vpc_name: str,
        env: cdk.Environment,
        **kwargs
    ) -> None:
        super().__init__(scope, id, env=env, **kwargs)

        self.prefix = self.node.try_get_context("resource_prefix")

        # Verificar si estamos en modo local
        local_mode = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

        # Lookup de la VPC usando el nombre pasado como argumento
        self.vpc = ec2.Vpc.from_lookup(self, id=f"{self.prefix}-VpcLookup", vpc_id=vpc_name)

        # Crear el rol de ejecución de SageMaker
        self.sm_execution_role = self.create_execution_role()

        if not local_mode:
            # Obtener el bucket de S3 para los datos de SageMaker desde el entorno o crearlo si no existe
            data_bucket_name = os.getenv("DATA_BUCKET")
            self.sm_data_bucket = self.get_or_create_bucket(data_bucket_name, "DataBucket")

            # Obtener el bucket de S3 para las fuentes de SageMaker desde el entorno o crearlo si no existe
            sources_bucket_name = os.getenv("SOURCES_BUCKET")
            self.sm_sources_bucket = self.get_or_create_bucket(sources_bucket_name, "SourcesBucket")

            # Conceder acceso de lectura/escritura al rol de ejecución de SageMaker en ambos buckets
            self.sm_data_bucket.grant_read_write(self.sm_execution_role)
            self.sm_sources_bucket.grant_read(self.sm_execution_role)

            # Crear los parámetros en SSM
            ssm.StringParameter(
                self, 'DataBucketName',
                parameter_name=f"/{self.prefix}/DataBucketName",
                string_value=self.sm_data_bucket.bucket_name,
                description="Data bucket name for SageMaker"
            )

            ssm.StringParameter(
                self, 'SourcesBucketName',
                parameter_name=f"/{self.prefix}/SourcesBucketName",
                string_value=self.sm_sources_bucket.bucket_name,
                description="Sources bucket name for SageMaker"
            )

            ssm.StringParameter(
                self, 'SagemakerExecutionRoleArn',
                parameter_name=f"/{self.prefix}/SagemakerExecutionRoleArn",
                string_value=self.sm_execution_role.role_arn,
                description="SageMaker Execution Role ARN"
            )

    def create_execution_role(self) -> iam.Role:
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

        # Agregar permisos específicos para los buckets de datos y fuentes
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            resources=[
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}/*",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}/*"
            ]
        ))

        return role

    def get_or_create_bucket(self, bucket_name: str, bucket_id: str) -> s3.Bucket:
        """
        Obtiene un bucket S3 por su nombre si existe, o lo crea si no.
        """
        if bucket_name:
            try:
                # Intentar reutilizar el bucket si ya existe
                return s3.Bucket.from_bucket_name(self, bucket_id, bucket_name=bucket_name)
            except Exception as e:
                print(f"El bucket '{bucket_name}' especificado no existe o no es accesible. Detalles del error: {e}")

        # Si el bucket no existe o no se especifica, crearlo
        print(f"Creando nuevo bucket con ID '{bucket_id}' ya que no se encontró '{bucket_name}' en el entorno.")
        return s3.Bucket(
            self,
            id=bucket_id,
            bucket_name=f"{self.prefix}-{bucket_id.lower()}-{self.account}",
            lifecycle_rules=[],
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
