import os
import re
import base64
import json
import yaml
import boto3
import awswrangler as wr
import logging

from dotenv import load_dotenv
from pipelines.common.utils.config import AWS_REGION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")


def running_in_sagemaker():
    return os.environ.get("SAGEMAKER_ENV") or os.path.exists("/opt/ml/processing/input")


def setup_boto_session(stage='dev'):
    """
    Configura la sesión boto3 si estás fuera de SageMaker.
    """
    profile_name = 'default' if stage == 'prod' else 'sandbox'
    boto3.setup_default_session(profile_name=profile_name, region_name=AWS_REGION)
    logger.info(f"🧑‍💻 Ejecutando localmente. Usando perfil local: {profile_name}")
    logger.info(f"Stage detectado en setup_boto_session: {stage}")


def read_from_athena(database, table, stage='dev', columns=None, filter_key=None,
                     filter_values=None, where_clause=None, chunksize=None, rename_dict=None,
                     read_from_prod=False):

    # 1. Si hay credenciales personalizadas
    if os.path.exists('./config_auth.yml'):
        with open('./config_auth.yml', 'r') as file:
            config = yaml.safe_load(file)
            boto3.setup_default_session(
                aws_access_key_id=config['aws_access_key_id'],
                aws_secret_access_key=config['aws_secret_access_key'],
                aws_session_token=config['aws_session_token'],
                region_name=config['region_name']
            )
        logger.info("🔐 Configuración con credenciales desde config_auth.yml")

    # 2. Si estás local
    elif not running_in_sagemaker():
        setup_boto_session(stage)

    # 3. Si estás en SageMaker
    else:
        boto3.setup_default_session(region_name=AWS_REGION)
        logger.info("🟢 Ejecutando dentro de SageMaker. Se configuró sesión boto3 con región.")

    GLUE_CLIENT = boto3.client('glue', region_name=AWS_REGION)
    passed_database = database
    original_columns = []

    if stage == 'dev' and read_from_prod:
        match = re.search(r'_v\d+', table)
        if match:
            table = table.split(match.group(0))[0]
        else:
            logger.warning(f"No se encontró versión en el nombre de la tabla: {table}. Se usará tal cual.")    

        # Cambiar el nombre de base de datos
        database = "prod_" + database
        logger.info(f"DEBUG - database: {database}, table: {table}, stage: {stage}, read_from_prod: {read_from_prod}")
        
        # Obtener columnas
        response = GLUE_CLIENT.get_table(DatabaseName=database, Name=table)
        original_columns_dict = {col['Name']: col['Type'] for col in response['Table']['StorageDescriptor']['Columns']}

        # Si es una vista, extraer tabla real
        if "ViewOriginalText" in response["Table"]:
            base64_string = response["Table"]["ViewOriginalText"]
            decode_me = base64_string[base64_string.index('/* Presto View: ') + len('/* Presto View: '):base64_string.index(' */')]
            table_sql_dict = json.loads(base64.b64decode(decode_me))
            original_sql = table_sql_dict['originalSql']
            table = re.search(r"([\w]+)$", original_sql).group(1)
        else:
            logger.warning('❗ La tabla no es una vista. No se puede extraer nombre original.')

        # Columnas finales
        if columns:
            intersect_columns_dict = {k: v for k, v in original_columns_dict.items() if k in columns}
        else:
            intersect_columns_dict = original_columns_dict

        if passed_database == "refined":
            original_columns = [f"CAST({k} AS timestamp(3)) as {k}" if "timestamp" in v else k for k, v in intersect_columns_dict.items()]
        else:
            original_columns = list(intersect_columns_dict.keys())

    else:
        original_columns = columns if columns else ['*']

    sql = f"SELECT {', '.join(original_columns)} FROM {database}.{table}"

    if filter_key and filter_values and where_clause:
        raise ValueError("❌ No puedes usar simultáneamente filter_key+values y where_clause")
    if filter_key and filter_values:
        sql += f" WHERE {filter_key} IN ({', '.join(map(str, filter_values))})"
    if where_clause:
        sql += f" {where_clause}"

    logger.info(f"🧪 Ejecutando query Athena: {sql}")

    df = wr.athena.read_sql_query(
        sql,
        database=passed_database,
        ctas_approach=False,
        chunksize=chunksize,
    )

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    return df