import os
import sys
import subprocess
import logging
import argparse
import pandas as pd
import numpy as np
import boto3

# Instalación del paquete en ejecución dentro de SageMaker
if os.path.exists("/opt/ml/processing/source_code"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "/opt/ml/processing/source_code"])
    sys.path.insert(0, "/opt/ml/processing/source_code")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports de lógica de negocio
from pipelines.common.api.athena import read_from_athena
from pipelines.lead_conversion_rate.common.utils.data_prep import (
    course_info_data_prep, contacts_info_data_prep,
    get_merged_contacts, get_deleted_contacts, get_merged_deals, get_deleted_deals,
    deals_to_course_data_prep, learn_deals_data_prep,
    contact_to_deals_data_prep, contact_analytics_data_prep,
    get_features, sanitize_string, format_duration
)
from pipelines.lead_conversion_rate.common.utils.feature_engineering import (
    Preprocess, cleanup_baseline_df, find_nearest_dumpdate,
    get_compare_date, sanitize_string, format_duration
)
from pipelines.lead_conversion_rate.common.constants import CLOSED_WIN


def read_data(env, pickle=False, target=True):
    baseline_df = get_features(stage=env)
    if target:
        if 'target' not in baseline_df.columns:
            baseline_df['target'] = 0
            baseline_df.loc[baseline_df['dealstage'].isin(CLOSED_WIN), 'target'] = 1
            baseline_df = Preprocess().remove_unneeded_cols(baseline_df)
    else:
        baseline_df['target'] = -1

    if pickle:
        baseline_df.to_pickle("./pickles/baseline_features_raw.pkl")

    return baseline_df



def save_data_to_output(data: pd.DataFrame):
    IN_SAGEMAKER = os.path.exists("/opt/ml/processing/input")
    output_dir = "/opt/ml/processing/output" if IN_SAGEMAKER else "./pickles"

    try:
        if not os.path.exists(output_dir):
            logger.warning(f"📁 El directorio '{output_dir}' no existe. Se crea...")
            os.makedirs(output_dir, exist_ok=True)

        pickle_path = os.path.join(output_dir, "test_output.pkl")
        logger.info(f"💾 Guardando Pickle en: {pickle_path}")
        data.to_pickle(pickle_path)
        logger.info(f"✅ Pickle guardado correctamente en: {pickle_path}")

        if os.path.exists(pickle_path):
            logger.info(f"📁 Archivo generado correctamente: {pickle_path}")
        else:
            logger.error("❌ Error: El archivo Pickle no fue creado.")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"❌ Excepción al guardar el Pickle: {e}")
        sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, required=True)
    args = parser.parse_args()
    env = args.environment

    print(f"✅ Ejecutando: {__file__}")
    print(f"🔧 Argumento recibido: env = {env}")
    logger.info(f"🧪 Entorno recibido: {env}")

    # Mostrar info del rol
    
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    logger.info(f"🔐 Rol en ejecución: {identity['Arn']}")

    try:
        data = read_data(env)
        logger.info("✅ Datos leídos correctamente. Mostrando primeras filas:")
        logger.info("\n" + data.head(10).to_string())
        save_data_to_output(data)

    except Exception as e:
        logger.exception(f"❌ Error durante la ejecución principal: {e}")
        sys.exit(1)

    logger.info("🏁 Script finalizado exitosamente.")
    sys.exit(0)
    
