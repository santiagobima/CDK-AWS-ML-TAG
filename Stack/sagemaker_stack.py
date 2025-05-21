import os
import logging
import aws_cdk as cdk
import re
import boto3
from aws_cdk import (
    aws_iam as iam,
    aws_s3 as s3,
    aws_ec2 as ec2,
    aws_ssm as ssm,
)
from constructs import Construct

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SagemakerStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        env: cdk.Environment,
        vpc_name: str,
        local_mode: bool,
        **kwargs
    ) -> None:
        super().__init__(scope, id, env=env, **kwargs)

        self.prefix = self.node.try_get_context("resource_prefix")

        # Lookup VPC
        self.vpc = ec2.Vpc.from_lookup(self, id=f"{self.prefix}-VpcLookup", vpc_id=vpc_name)
        logger.info(f"VPC '{vpc_name}' cargada exitosamente.")

        # Create IAM role for SageMaker
        self.sm_execution_role = self.create_execution_role()

        # Buckets
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

        self.sm_data_bucket.grant_read_write(self.sm_execution_role)
        self.sm_sources_bucket.grant_read(self.sm_execution_role)

        # Create SSM Parameters
        self.create_ssm_parameters()

    def create_execution_role(self) -> iam.Role:
        role_name = f"{self.prefix}-sm-execution-role"
        role_name = re.sub(r"[^a-zA-Z0-9+=,.@_-]", "", role_name)[:64].strip("_-.")

        logger.info(f"Generando rol con nombre: {role_name}")

        role = iam.Role(
            self, 'SagemakerExecutionRole',
            assumed_by=iam.ServicePrincipal('sagemaker.amazonaws.com'),
            role_name=role_name,
            managed_policies=[
                iam.ManagedPolicy.from_managed_policy_arn(
                    self,
                    id="SagemakerFullAccess",
                    managed_policy_arn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                ),
            ],
        )

        logger.info("Rol de ejecución de SageMaker creado con éxito.")

        # Permisos adicionales
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "lakeformation:*"
            ],
            resources=["*"]
        ))

        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:GetObject", "s3:PutObject", "s3:DeleteObject"
            ],
            resources=[
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}/*",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}/*",
                "arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1",
                "arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1/*"
            ]
        ))

        role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}",
                "arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1",
            ]
        ))

        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:GetWorkGroup"
            ],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
        ))

        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "glue:GetTable", "glue:GetDatabase", "glue:GetPartition"
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/{os.getenv('DATABASE')}",
                f"arn:aws:glue:{self.region}:{self.account}:table/{os.getenv('DATABASE')}/*"
            ]
        ))

        return role

    def get_or_create_bucket(self, bucket_name: str, bucket_id: str, description: str) -> s3.Bucket:
        if bucket_name:
            try:
                logger.info(f"Intentando cargar bucket '{bucket_name}'.")
                return s3.Bucket.from_bucket_name(self, bucket_id, bucket_name=bucket_name)
            except Exception as e:
                logger.warning(f"Bucket '{bucket_name}' no encontrado. Error: {e}")

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
        ssm_client = boto3.client("ssm", region_name=self.region)

        def parameter_exists(parameter_name):
            try:
                ssm_client.get_parameter(Name=parameter_name)
                return True
            except ssm_client.exceptions.ParameterNotFound:
                return False

        parameters = {
            "DataBucketName": self.sm_data_bucket.bucket_name,
            "SourcesBucketName": self.sm_sources_bucket.bucket_name,
            "SagemakerExecutionRoleArn": self.sm_execution_role.role_arn
        }

        for param_name, param_value in parameters.items():
            full_param_name = f"/{self.prefix}/{param_name}"
            if not parameter_exists(full_param_name):
                ssm.StringParameter(
                    self, param_name,
                    parameter_name=full_param_name,
                    string_value=param_value,
                    description=f"{param_name} for SageMaker"
                )
                logger.info(f"Parámetro SSM '{full_param_name}' creado.")
            else:
                logger.info(f"Parámetro SSM '{full_param_name}' ya existe. No se crea nuevamente.")