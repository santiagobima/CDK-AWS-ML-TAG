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

        # Lookup de la VPC usando el nombre pasado como argumento
        self.vpc = ec2.Vpc.from_lookup(self, id=f"{self.prefix}-VpcLookup", vpc_id=vpc_name)

        # Crear el rol de ejecución de SageMaker
        self.sm_execution_role = self.create_execution_role()

        # Crear el bucket de S3 para los fuentes de SageMaker
        self.sm_sources_bucket = self.create_sm_sources_bucket()

        ssm.StringParameter(
            self, 'SourcesBucketName',
            string_value=self.sm_sources_bucket.bucket_name,
            parameter_name=f"/{self.prefix}/SourcesBucketName",
            description="SageMaker Sources Bucket Name",
        )

        # Conceder acceso de lectura al rol de ejecución de SageMaker
        self.sm_sources_bucket.grant_read(self.sm_execution_role)

        # Crear el bucket de S3 para los datos de SageMaker
        self.sm_data_bucket = self.create_data_bucket()

        # Conceder acceso de lectura/escritura al rol de ejecución de SageMaker
        self.sm_data_bucket.grant_read_write(self.sm_execution_role)

        # Obtener los subnets públicos de la VPC
        public_subnet_ids = [public_subnet.subnet_id for public_subnet in self.vpc.public_subnets]

        # Crear el dominio de SageMaker Studio
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

    # Métodos auxiliares (sin cambios)
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
        ssm.StringParameter(
            self, 'SagemakerExecutionRoleArn',
            string_value=role.role_arn,
            parameter_name=f"/{self.prefix}/SagemakerExecutionRoleArn",
            description="SageMaker Execution Role ARN",
        )

        return role

    def create_sm_sources_bucket(self) -> s3.Bucket:
        return s3.Bucket(
            self,
            id="SourcesBucket",
            bucket_name=f"{self.prefix}-sm-sources",
            lifecycle_rules=[],
            versioned=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            # Access
            access_control=s3.BucketAccessControl.PRIVATE,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            public_read_access=False,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            enforce_ssl=True,
            # Encryption
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

    def create_data_bucket(self):
        return s3.Bucket(
            self,
            id="DataBucket",
            bucket_name=f"{self.prefix}-sm-data",
            lifecycle_rules=[],
            versioned=False,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            # Access
            access_control=s3.BucketAccessControl.PRIVATE,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            public_read_access=False,
            object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
            enforce_ssl=True,
            # Encryption
            encryption=s3.BucketEncryption.S3_MANAGED,
        )