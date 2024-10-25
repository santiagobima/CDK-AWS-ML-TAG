import os
import aws_cdk as cdk
from aws_cdk import (
    aws_sagemaker as sm,
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
        vpc_name: str,  # Pasamos el nombre de la VPC en lugar de un objeto VPC
        env: cdk.Environment,  # Env se mantiene igual
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
            # Crear o reutilizar el bucket de S3 para los fuentes de SageMaker
            self.sm_sources_bucket = self.create_sm_sources_bucket()

            # Verificar si el parámetro ya existe en SSM y crearlo si no existe
            self.ensure_ssm_parameter(
                name=f"/{self.prefix}/SourcesBucketName",
                value=self.sm_sources_bucket.bucket_name,
                description="SageMaker Sources Bucket Name"
            )

            # Conceder acceso de lectura al rol de ejecución de SageMaker
            self.sm_sources_bucket.grant_read(self.sm_execution_role)

            # Crear o reutilizar el bucket de S3 para los datos de SageMaker
            self.sm_data_bucket = self.create_data_bucket()

            # Conceder acceso de lectura/escritura al rol de ejecución de SageMaker
            self.sm_data_bucket.grant_read_write(self.sm_execution_role)

            # Obtener los subnets públicos de la VPC
            public_subnet_ids = [public_subnet.subnet_id for public_subnet in self.vpc.public_subnets]

            # Crear el dominio de SageMaker Studio (solo en modo no local)
            self.domain = sm.CfnDomain(
                self, "SagemakerDomain",
                auth_mode='IAM',
                domain_name=f'{self.prefix}-SG-Project',
                default_user_settings=sm.CfnDomain.UserSettingsProperty(
                    execution_role=self.sm_execution_role.role_arn
                ),
                app_network_access_type='PublicInternetOnly',
                vpc_id=self.vpc.vpc_id,
                subnet_ids=public_subnet_ids,
                tags=[cdk.CfnTag(
                    key="project",
                    value="example-pipelines"
                )],
            )

            # Crear el perfil de usuario predeterminado de SageMaker Studio
            self.user = sm.CfnUserProfile(
                self, 'SageMakerStudioUserProfile',
                domain_id=self.domain.attr_domain_id,
                user_profile_name='default-user',
                user_settings=sm.CfnUserProfile.UserSettingsProperty(),
            )

    def ensure_ssm_parameter(self, name: str, value: str, description: str):
        """
        Verificar si un parámetro ya existe en SSM y crearlo si no está presente.
        """
        try:
            existing_parameter = ssm.StringParameter.from_string_parameter_name(
                self, 'ExistingParameter', string_parameter_name=name)
            print(f"Parámetro existente encontrado: {existing_parameter.string_value}")
        except Exception:
            print(f"Parámetro no encontrado, creando: {name}")
            ssm.StringParameter(
                self, 'NewParameter',
                string_value=value,
                parameter_name=name,
                description=description,
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
        
        # Almacenar el ARN del rol en SSM
        self.ensure_ssm_parameter(
            name=f"/{self.prefix}/SagemakerExecutionRoleArn",
            value=role.role_arn,
            description="SageMaker Execution Role ARN"
        )

        return role

    def create_sm_sources_bucket(self) -> s3.Bucket:
        try:
            # Intentar reutilizar el bucket si ya existe
            return s3.Bucket.from_bucket_name(self, "ExistingSourcesBucket", bucket_name=f"{self.prefix}-sm-sources")
        except Exception as e:
            # Si no existe, crear uno nuevo
            print(f"El bucket no existe, creando uno nuevo. Detalles del error: {e}")
            return s3.Bucket(
                self,
                id="SourcesBucket",
                bucket_name=f"{self.prefix}-sm-sources-{self.account}",  # Cambia el nombre para que sea único
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

    def create_data_bucket(self) -> s3.Bucket:
        # Crear un nuevo bucket de datos siempre, con un nombre único basado en el prefijo y la cuenta
        return s3.Bucket(
            self,
            id="DataBucket",
            bucket_name=f"{self.prefix}-sm-data-{self.account}",  # Asegura un nombre único para el bucket
            lifecycle_rules=[],
            versioned=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,  # Eliminar el bucket al destruir la stack
            auto_delete_objects=True,  # Borrar los objetos al eliminar el bucket
            access_control=s3.BucketAccessControl.PRIVATE,  # Acceso privado al bucket
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,  # Bloquear acceso público
            public_read_access=False,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED,  # Encriptar los objetos en el bucket
        )
