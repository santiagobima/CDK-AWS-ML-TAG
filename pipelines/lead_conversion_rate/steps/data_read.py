import os
import sys
import subprocess
import logging
import boto3

# Instalar directamente como paquete desde la carpeta descomprimida
if os.path.exists("/opt/ml/processing/source_code"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "/opt/ml/processing/source_code"])
    sys.path.insert(0, "/opt/ml/processing/source_code")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import argparse

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

def save_data(data, data_path):
    data.to_pickle(data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        required=True
    )
    args=parser.parse_args()
    env = args.environment
    logger.info(f" Environment is = {env}")
    print(f"✅ Ejecutando: {__file__}")
    print(f"🔧 Argumento recibido: env = {env}")
    
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    logger.info(f"🔐 Rol en ejecución:")
    logger.info(f"  ARN: {identity['Arn']}")
    logger.info(f"  Cuenta: {identity['Account']}")
    logger.info(f"  Usuario: {identity['UserId']}")
    
    data = read_data(env)
    logger.info("✅ Ejecución completada.")
    logger.info("📊 Primeras filas:")
    logger.info(data.head(10).to_string())
    
    output_path = "/opt/ml/processing/output/test_output.csv"
    data.head(10).to_csv(output_path, index=False)
    
"""import boto3
import os

region = os.getenv("AWS_REGION", "eu-west-1")  # Fallback por si no está definido

print("🚀 Empezando test simple de acceso a Glue desde SageMaker")

sts = boto3.client("sts", region_name=region)
identity = sts.get_caller_identity()
print(f"🔐 Rol en ejecución (ARN): {identity['Arn']}")

glue = boto3.client("glue", region_name=region)
response = glue.get_table(
    DatabaseName="prod_refined",
    Name="hubspot_deals_stage_support_latest"
)
print(f"✅ Tabla encontrada: {response['Table']['Name']}")"""