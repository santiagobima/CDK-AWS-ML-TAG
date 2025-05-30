import os
import sys
import subprocess
import logging

# Instalar directamente como paquete desde la carpeta descomprimida
if os.path.exists("/opt/ml/processing/source_code"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "/opt/ml/processing/source_code"])
    sys.path.insert(0, "/opt/ml/processing/source_code")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

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

def read_data(pickle=False, target=True):
    baseline_df = get_features()
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
    data = read_data()
    logger.info("✅ Ejecución completada.")
    logger.info("📊 Primeras filas:")
    logger.info(data.head(10).to_string())

    output_path = "/opt/ml/processing/output/test_output.csv"
    data.head(10).to_csv(output_path, index=False)