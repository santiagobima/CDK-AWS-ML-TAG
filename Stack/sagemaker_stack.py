import os
import logging
import aws_cdk as cdk
import re
import boto3
import shutil
from pathlib import Path
from aws_cdk import aws_iam as iam, aws_s3 as s3, aws_ec2 as ec2, aws_ssm as ssm, aws_lakeformation as lakeformation, aws_ecr_assets as ecr_assets
from constructs import Construct
from aws_cdk.aws_ecr_assets import Platform

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
        
        # Definir rutas
        sources_dir = Path("Pipelines/lead_conversion_rate/sources/")
        image_dir = Path("image/")

        # Asegurar que la carpeta `image/` existe
        image_dir.mkdir(parents=True, exist_ok=True)

        # Copiar automáticamente TODOS los archivos Python de `sources/` a `image/`
        #for file in sources_dir.glob("*.py"):
        #    shutil.copy(file, image_dir / file.name)
        #    print(f"📂 Copiado: {file} -> {image_dir / file.name}")
         
             
        self.ecr_image = ecr_assets.DockerImageAsset(
            self,
            "sagemakerPipelineImage",
            directory = ".",
            file= 'image/Dockerfile',
            platform=Platform.LINUX_AMD64,
            
            
        )
        
        self.image_uri = self.ecr_image.image_uri
        logger.info(f'Imagen de ECR creada en: {self.image_uri}')
            

        # Crear parámetros en SSM
        self.create_ssm_parameters()

    def create_execution_role(self) -> iam.Role:
        """
        Crea el rol de ejecución de SageMaker con los permisos necesarios.

        :return: El rol de IAM creado.
        """
                
        role_name = f"{self.prefix}-sm-execution-role"

        role_name = re.sub(r"[^a-zA-Z0-9+=,.@_-]", "", role_name)  

        # Asegurar que no exceda 64 caracteres y eliminar caracteres inválidos al inicio o final
        role_name = role_name[:64].strip("_-.")

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

        #  LAKEFORMATION IAM
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "lakeformation:*"
            ],
            resources=["*"]
        ))

        #  S3 IAM
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            resources=[
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}/*",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}/*",
                "arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1",
                f"arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1/*"
            ]
        ))

        # S3 List
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:ListBucket"
            ],
            resources=[
                f"arn:aws:s3:::{os.getenv('DATA_BUCKET')}",
                f"arn:aws:s3:::{os.getenv('SOURCES_BUCKET')}",
                "arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1",
                f"arn:aws:s3:::aws-athena-query-results-373024328391-eu-west-1/*"
            ]
        ))

        #  ATHENA IAM
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:GetWorkGroup"
            ],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
        ))

        #  GLUE IAM
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "glue:GetTable",
                "glue:GetDatabase",
                "glue:GetPartition",

            ],   
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/{os.getenv('DATABASE')}",
                f"arn:aws:glue:{self.region}:{self.account}:table/{os.getenv('DATABASE')}/*"
            ]
        ))

        """lakeformation.CfnPermissions(self, "SagemakerLakeformationPermission",
            data_lake_principal=lakeformation.CfnPermissions.DataLakePrincipalProperty(
                data_lake_principal_identifier=role.role_arn
            ),
            resource=lakeformation.CfnPermissions.ResourceProperty(
                database_resource=lakeformation.CfnPermissions.DatabaseResourceProperty(
                    catalog_id=os.getenv("CDK_DEFAULT_ACCOUNT"),
                    name="refined"
                ),
                table_resource=lakeformation.CfnPermissions.TableResourceProperty(
                    catalog_id=os.getenv("CDK_DEFAULT_ACCOUNT"),
                    database_name="refined",
                    table_wildcard=lakeformation.CfnPermissions.TableWildcardProperty()
                )
            )
        ) """

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
        y el ARN del rol de ejecución solo si no existen.
        """
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
            "SagemakerExecutionRoleArn": self.sm_execution_role.role_arn,
            "PipelineImageUri": self.image_uri
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
