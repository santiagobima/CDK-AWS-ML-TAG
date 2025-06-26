import boto3
import pandas as pd
import io
import logging

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parámetros
BUCKET_NAME = "tag-dl-sandbox-data"
S3_KEY = "output-data/test_output.pkl"
LOCAL_PICKLE_PATH = "./pickles/baseline_features_raw.pkl"
ID_FIELD = "contact_id"

# Función para leer Pickle desde S3
def read_pickle_from_s3(bucket: str, key: str) -> pd.DataFrame:
    logger.info("📥 Leyendo Pickle desde S3...")
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_pickle(io.BytesIO(response["Body"].read()))
    logger.info("✅ Pickle desde S3 cargado correctamente.")
    return df

# Función para leer Pickle local
def read_local_pickle(path: str) -> pd.DataFrame:
    logger.info("📥 Leyendo Pickle local...")
    df = pd.read_pickle(path)
    logger.info("✅ Pickle local cargado correctamente.")
    return df

# Función para detectar duplicados
def find_duplicates(df: pd.DataFrame, key: str, label: str):
    dupes = df[df.duplicated(subset=[key], keep=False)]
    logger.info(f"🔁 Duplicados en {label}: {len(dupes)} filas, {dupes[key].nunique()} IDs únicos")
    if not dupes.empty:
        logger.info(f"🔁 Ejemplos de duplicados:\n{dupes[[key]].value_counts().head()}")

# Comparación completa
def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, key: str):
    logger.info("🔍 Comparando columnas...")
    if list(df1.columns) != list(df2.columns):
        logger.warning("⚠️ Las columnas son diferentes.")
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        logger.info(f"Solo en S3: {cols1 - cols2}")
        logger.info(f"Solo en Local: {cols2 - cols1}")
    else:
        logger.info("✅ Las columnas son iguales.")

    logger.info("🔍 Buscando IDs comunes...")
    common_ids = set(df1[key]) & set(df2[key])
    logger.info(f"🔢 Contactos en común: {len(common_ids)}")

    # Ver duplicados
    find_duplicates(df1, key, label="S3")
    find_duplicates(df2, key, label="Local")

    # Filtrar y limpiar
    df1_common = df1[df1[key].isin(common_ids)].drop_duplicates(key).sort_values(key).reset_index(drop=True)
    df2_common = df2[df2[key].isin(common_ids)].drop_duplicates(key).sort_values(key).reset_index(drop=True)

    logger.info("🧪 Comparando registros únicos por contact_id...")
    try:
        pd.testing.assert_frame_equal(df1_common, df2_common, check_dtype=False)
        logger.info("🎯 ¡Los DataFrames filtrados son IGUALES!")
    except AssertionError as e:
        logger.error("❌ Diferencias encontradas entre los registros comunes:")
        logger.error(str(e))


if __name__ == "__main__":
    df_s3 = read_pickle_from_s3(BUCKET_NAME, S3_KEY)
    df_local = read_local_pickle(LOCAL_PICKLE_PATH)
    compare_dataframes(df_s3, df_local, key=ID_FIELD)