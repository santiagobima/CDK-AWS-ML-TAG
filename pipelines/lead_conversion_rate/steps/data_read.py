import os
import sys

# Agregar la raíz del proyecto a sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import pandas as pd
import numpy as np
import logging

from Pipelines.common.api.athena import read_from_athena
from Pipelines.lead_conversion_rate.common.utils.data_prep import (
    course_info_data_prep, 
    contacts_info_data_prep, 
    get_merged_contacts, 
    get_deleted_contacts, 
    get_merged_deals, 
    get_deleted_deals,
    deals_to_course_data_prep,
    learn_deals_data_prep,
    contact_to_deals_data_prep,
    contact_analytics_data_prep
)

from Pipelines.lead_conversion_rate.common.utils.feature_engineering import (
    Preprocess,
    cleanup_baseline_df,
    find_nearest_dumpdate,
    get_compare_date,
    sanitize_string,
    format_duration
)

from Pipelines.lead_conversion_rate.common.utils.feature_engineering import (
    Preprocess,
    cleanup_baseline_df,
    find_nearest_dumpdate,
    get_compare_date,
    sanitize_string,
    format_duration
)

from Pipelines.lead_conversion_rate.common.constants import CLOSED_WIN
from Pipelines.lead_conversion_rate.common.utils.data_prep import get_features
from Pipelines.lead_conversion_rate.common.utils.data_prep import sanitize_string, format_duration

def read_data(pickle=False, target=True):
    """
    Read or generate the baseline DataFrame.

    Parameters:
        pickle (bool): If True, save the data to a pickle file for local use.
        local_source (bool): If True, read data locally from `data_path`. If False, generate
        features.
        data_path (str): Path to the pickle file or data source.
        target (bool): If True, ensure the DataFrame has a 'target' column based on 'dealstage'.

    Returns:
        pd.DataFrame: The baseline DataFrame with or without target column.
    """
    
    baseline_df = get_features()  # Assuming get_features() is a function to generate data

    if target:
        if 'target' not in baseline_df.columns:
            baseline_df['target'] = 0
            baseline_df.loc[baseline_df['dealstage'].isin(CLOSED_WIN), 'target'] = 1
            baseline_df = Preprocess().remove_unneeded_cols(baseline_df)

        if pickle:
            baseline_df.to_pickle("./pickles/baseline_features_raw.pkl")
    else:
        baseline_df['target'] = -1

    return baseline_df


def save_data(data, data_path, local_source=True):
    """
    Save the data

    Parameters:
        data (pd.DataFrame): The DataFrame to save.
        data_path (str): Path to save the pickle file.
        local_source (bool): If True, save the pickle file locally.

    """
    if local_source:
        data.to_pickle(data_path)


if __name__ == "__main__":
    data = read_data(
        local_source=False,
        data_path="./pickles/new_baseline_features_raw.pkl"
    )
    
    # Puedes imprimir algo para verificar que se ejecuta correctamente
    print("Ejecución completada. Datos cargados correctamente.")