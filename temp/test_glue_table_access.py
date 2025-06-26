import boto3
import logging
import os
from botocore.exceptions import ClientError

# Configura el entorno
os.environ["AWS_PROFILE"] = "sandbox"
os.environ["AWS_REGION"] = "eu-west-1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parámetros de prueba
catalog_name = "prod_refined"   # <- Este es el que falla
table_name = "hubspot_deals_stage_support_latest"

# Crea un cliente de Glue
glue = boto3.client("glue", region_name=os.environ["AWS_REGION"])

logger.info("🔍 Intentando acceder a Glue con nombre de catálogo: %s", catalog_name)

try:
    response = glue.get_table(DatabaseName=catalog_name, Name=table_name)
    logger.info("✅ Éxito. Se encontró la tabla:")
    print(response["Table"]["Name"])
except ClientError as e:
    logger.error("❌ Error al acceder a Glue:")
    logger.error(e.response["Error"]["Message"])
    logger.error("🔐 Rol en uso: %s", boto3.client("sts").get_caller_identity()["Arn"])