import os
import yaml
import json
import base64
import boto3
import awswrangler as wr
import logging
import re
import pandas as pd
from dotenv import load_dotenv

# Cargar variables desde env/.env.dev
dotenv_path = os.path.join(os.path.dirname(__file__), '../../env/.env.dev')
load_dotenv(dotenv_path)

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Leer variables de entorno
DATA_BUCKET = os.getenv("DATA_BUCKET")
SOURCES_BUCKET = os.getenv("SOURCES_BUCKET")
DATABASE = os.getenv("DATABASE")
TABLE = os.getenv("TABLE")

# Validar que las variables existen
if not DATA_BUCKET:
    raise ValueError("ERROR: No se ha encontrado la variable de entorno 'DATA_BUCKET'")
if not SOURCES_BUCKET:
    raise ValueError("ERROR: No se ha encontrado la variable de entorno 'SOURCES_BUCKET'")

def read_from_athena(database, table, stage='dev', columns=None, filter_key=None,
                     filter_values=None, where_clause=None, chunksize=None, rename_dict=None,
                     read_from_prod=False):
    """
    Lee datos desde AWS Athena y devuelve un DataFrame.
    """
    
    REGION = os.getenv("CDK_DEFAULT_REGION", "eu-west-1")

    # Autenticación con AWS
    if os.path.exists('./config_auth.yml'):
        with open('./config_auth.yml', 'r') as file:
            config = yaml.safe_load(file)
            aws_access_key_id = config['aws_access_key_id']
            aws_secret_access_key = config['aws_secret_access_key']
            aws_session_token = config['aws_session_token']
            region_name = config['region_name']

        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token, region_name=region_name
        )
    else:
        boto3.setup_default_session(region_name=REGION)

    GLUE_CLIENT = boto3.client('glue', region_name=REGION)

    passed_database = database
    original_columns = []

    if stage == 'dev' and read_from_prod:
        table = table.split(re.search(r'_v\d+', table).group(0))[0]

        database = "prod_" + database
        response = GLUE_CLIENT.get_table(DatabaseName=database, Name=table)
        original_columns_dict = {col['Name']: col['Type'] for col in
                                 response['Table']['StorageDescriptor']['Columns']}

        base64_string = response["Table"]["ViewOriginalText"]
        decode_me = base64_string[base64_string.index('/* Presto View: ') + len(
            '/* Presto View: '):base64_string.index(' */')]
        table_sql_dict = json.loads(base64.b64decode(decode_me))
        original_sql = table_sql_dict['originalSql']
        table = re.search(r"([\w]+)$", original_sql).group(1)

        if columns:
            intersect_columns = [col for col in columns if col in original_columns_dict.keys()]
            intersect_columns_dict = {k: v for k, v in original_columns_dict.items() if
                                      k in intersect_columns}
        else:
            intersect_columns_dict = original_columns_dict

        if passed_database == "refined":
            original_columns = [f"CAST ({k} AS timestamp(3)) as {k}" if "timestamp" in v else k for
                                k, v in intersect_columns_dict.items()]
    else:
        original_columns = columns if columns else ['*']

    sql = f"SELECT {', '.join(original_columns)} FROM {database}.{table}"

    if filter_key and filter_values and where_clause:
        raise ValueError("You can only use one of filter_key/filter_values or where_clause")
    if filter_key and filter_values:
        sql += f" WHERE {filter_key} IN ({', '.join(list(map(str, filter_values)))})"
    if where_clause:
        sql += f" {where_clause}"

    logger.info(f"Ejecutando consulta: {sql}")

    df = wr.athena.read_sql_query(
        sql,
        database=passed_database,
        ctas_approach=False,
        chunksize=chunksize,
    )

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    return df

def save_to_s3(df: pd.DataFrame, bucket: str, path: str):
    """
    Guarda un DataFrame como CSV en S3 usando awswrangler.
    """
    s3_path = f"s3://{bucket}/{path}"
    logger.info(f"Intentando guardar en {s3_path}")

    try:
        wr.s3.to_csv(df, path=s3_path, index=False)
        logger.info(f"Archivo guardado en {s3_path}")
    except Exception as e:
        logger.error(f"Error al guardar archivo en S3: {e}")
        raise

def read_from_s3(bucket: str, path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV desde S3 y lo devuelve como un DataFrame.
    """
    s3_path = f"s3://{bucket}/{path}"
    logger.info(f"Intentando leer desde {s3_path}")

    try:
        df = wr.s3.read_csv(path=s3_path)
        logger.info(f"Archivo cargado desde {s3_path}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar archivo desde S3: {e}")
        raise

if __name__ == "__main__":
    try:
        df = read_from_athena(DATABASE, TABLE)
        print("Consulta ejecutada correctamente. Primeras filas:")
        print(df.head())

        print("Guardando archivo en S3...")
        test_path = "test-folder/data_test.csv"
        save_to_s3(df, DATA_BUCKET, test_path)

        print("Cargando archivo desde S3...")
        df_s3 = read_from_s3(DATA_BUCKET, test_path)
        print("Archivo cargado correctamente. Primeras filas:")
        print(df_s3.head())

    except Exception as e:
        print(f"Error en la ejecución: {e}")
        
        
        
