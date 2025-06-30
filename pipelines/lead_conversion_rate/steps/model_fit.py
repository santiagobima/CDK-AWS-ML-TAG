import os
import sys
import subprocess
import pandas as pd
import logging
import argparse


# Ensure dependencies and correct sys.path when running in SageMaker
if os.path.exists("/opt/ml/processing/source_code"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "/opt/ml/processing/source_code"])
    sys.path.insert(0, "/opt/ml/processing/source_code")

from pipelines.lead_conversion_rate.steps.data_read import read_data
from pipelines.lead_conversion_rate.steps.data_prep import preprocessing_pipeline
from pipelines.lead_conversion_rate.model.model import Model
from pipelines.lead_conversion_rate.model.utilities import (
    load_model, save_models, save_features, write_prediction, load_features
)
from pipelines.lead_conversion_rate.model.utls.utls import config, logger
from pipelines.lead_conversion_rate.common.utils.transformers import (
    BooleanTransformer, ReplaceTransformer, CountryCodeTransformer,
    LocationTransformer, EnrichmentTransformer, FillnaTransformer,
    DealCookingStateTransformer, ChangeTypeTransformer,
    YearsOfExperienceTransformer, ScalerTransformer,
    CombineProfileTransformer, OneHotEncodeTransformer,
    OneHotEncodeMultipleChoicesTransformer,
    CalculateTimeSinceTransformer, DropColumnsTransformer,
    FeatureNamesSanitizerTransformer, PreprocessSummary
)
from sklearn.pipeline import Pipeline
from pipelines.lead_conversion_rate.common.constants import (
    BOOLEAN_COLUMNS, REPLACE_DICT, ENRICHMENT_COLUMNS,
    FILLNA_VALUES, TYPE_DICT, COLUMNS_TO_SCALE,
    ONEHOT_COLUMNS, MULTIPLE_CATEGORIES, TIME_FIELDS
)
from pipelines.lead_conversion_rate.model.utls.utls import config


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)








def preprocessing_pipeline(prediction=True):
    steps = [
        ('boolean', BooleanTransformer(columns=BOOLEAN_COLUMNS)),
        ('replace', ReplaceTransformer(replace_dict=REPLACE_DICT)),
        ('convert_country', CountryCodeTransformer()),
        ('location', LocationTransformer()),
        ('enrich', EnrichmentTransformer(column_pairs=ENRICHMENT_COLUMNS)),
        ('fillna', FillnaTransformer(fill_values=FILLNA_VALUES)),
        ("deal_cooking", DealCookingStateTransformer()),
        ('replace_review_analysis', ReplaceTransformer(replace_dict={'review_analysis': {'no': '0', 'yes': '1'}})),
        ("type", ChangeTypeTransformer(TYPE_DICT)),
        ("years_of_exp_type", YearsOfExperienceTransformer()),
        ('scale', ScalerTransformer(columns=COLUMNS_TO_SCALE)),
        ('combine_profile', CombineProfileTransformer()),
        ('onehot_encode', OneHotEncodeTransformer(columns=ONEHOT_COLUMNS)),
        ('onehot_encode_multichoice', OneHotEncodeMultipleChoicesTransformer(columns=MULTIPLE_CATEGORIES)),
        ('time_since', CalculateTimeSinceTransformer(time_fields=TIME_FIELDS)),
        ('drop_columns', DropColumnsTransformer()),
        ("type_phase2", ChangeTypeTransformer()),
        ('name_sanitizer', FeatureNamesSanitizerTransformer())
    ]
    if not prediction:
        steps.append(('Summary', PreprocessSummary()))
    return Pipeline(steps=steps, verbose=True)




def predict(stage, data=None, transform=True):
    """
    Predict the outcomes for the given stage and data.

    Args:
        transform: If the data need to be transformed first
        stage (str): The stage of the prediction pipeline.
        data (dict, optional): The data to make predictions on.
                                 If None, data is read from the configured source.
                                 If dict, it will be converted to a DataFrame.

    Returns:
        pd.Series: The predictions.
    """

    # Read Data
    if data is None:
        data = read_data(local_source=config['Read']['data_source'].get('local'),
                         target=False,
                         data_path=config['Read']['data_source'].get('predict_data_path'))
    else:
        try:
            data = pd.DataFrame.from_records(data)
        except Exception as e:
            logger.error(f"Error: Data must be a list of dicts or dict-like. {e}")
            return None

    if transform:
        data = preprocessing_pipeline().transform(data)

    # Load features for the specified stage
    features_and_dtypes = load_features(config['Model'].get('name'), stage)
    features = features_and_dtypes.keys()
    missed_features = set(features) - set(data.columns)
    for feature in missed_features:
        data[feature] = 0

    data = data[features]
    # Convert the dtypes back to the appropriate format
    dtypes = {col: pd.api.types.pandas_dtype(dtype) for col, dtype in features_and_dtypes.items()}
    data = data.astype(dtypes)

    # Load model and predict probabilities
    prediction = load_model(config['Model'].get('name'), stage).predict_proba(data)[:, 1]

    # Write prediction
    return write_prediction(prediction, name=config['Model'].get('name'), stage=stage)

def fit():
    """
    Fit the model using the transformed old data.

    This function reads the transformed old data, fits the model, saves the best models,
    and saves the selected features.
    """
    # Read transformed old data
    # data = read_data(local_source=config['Read']['data_source'].get('local'),
    #                  data_path=config['Read']['data_source'].get(
    #                      'training_data_path'))

    # transformed_data = preprocessing_pipeline().fit_transform(data)
    
    IN_SAGEMAKER = os.path.exists('/opt/ml/processing/input')
    processed_data_path = "/opt/ml/processing/predict_input_data/baseline_features_raw.pkl" if IN_SAGEMAKER else "pipelines/lead_conversion_rate/model/pickles/baseline_features_raw.pkl"
    
    if not os.path.exists(processed_data_path):
        logger.error(f'File not found: {processed_data_path}')
        sys.exit(1)

    logger.info(f'Cargando datos procesados desde {processed_data_path}')
    
    transformed_data = pd.read_pickle(processed_data_path)        
    
    # Initialize model and fit
    model = Model()
    best_models, features = model.fit(transformed_data)

    # Save the best models and features
    save_models(best_models, model.name)
    save_features(transformed_data, features, model.name)

if __name__ == "__main__":
    fit()
