import os
import sys
import subprocess
import pandas as pd
import logging
import argparse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



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
from pipelines.lead_conversion_rate.model.utls.utls import logger
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
        env = os.getenv('ENV', 'dev')
        logger.info(f'Using ENV: {env}')
        print(f"🌍 Using ENV: {env}")
        data = read_data(env=env, local_source=config['Read']['data_source'].get('local'),
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
    
    print(">>>> INICIO DE FIT")
    try:
        env = os.getenv('ENV', 'dev')
        logger.info(f'Using ENV: {env}')
        print(f"🌍 Using ENV: {env}")

        data_path = config['Read']['data_source'].get('training_data_path')
        logger.info(f'Reading from {data_path}')
        print(f"📥 Reading from {data_path}")

        data = pd.read_pickle("temp/fit_sb/train.pkl")
        print(f"✅ Data loaded. Shape: {data.shape}")

        transformed_data = preprocessing_pipeline().fit_transform(data)
        print(f"✅ Data preprocessed. Shape: {transformed_data.shape}")

        model = Model()
        print("⚙️ Starting training...")
        best_models, features = model.fit(transformed_data)
        print("✅ Training complete.")

        save_models(best_models, model.name)
        print("✅ Models saved.")
        save_features(transformed_data, features, model.name)
        print("✅ Features saved.")
    except Exception as e:
        print('EROR IN FIT', e)
        
    print("🏁 END OF SCRIPT")

if __name__ == "__main__":
    fit()
