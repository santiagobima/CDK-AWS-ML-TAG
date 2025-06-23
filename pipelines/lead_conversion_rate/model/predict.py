import pandas as pd

#from data_io.features import read_data, save_data
from pipelines.lead_conversion_rate.steps.data_read import read_data
#from data_prep.preprocess import preprocessing_pipeline
from pipelines.lead_conversion_rate.steps.data_prep import preprocessing_pipeline
from model.model import Model
from model.utls import load_model, save_models, save_features, write_prediction, load_features
from pipelines.lead_conversion_rate.model.utls.utls import config, logger
import os
import sys


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
        if not isinstance(data[0], dict):
            logger.error("Error: Data must be a dictionary.")
            return None
        data = pd.DataFrame(data)

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
    #data = read_data(local_source=config['Read']['data_source'].get('local'),
    #                 data_path=config['Read']['data_source'].get(
     #                    'training_data_path'))

    #transformed_data = preprocessing_pipeline().fit_transform(data)
    
    IN_SAGEMAKER = os.path.exists('/opt/ml/processing/input')
    processed_data_path = "/opt/ml/processing/input/data_processed.pkl" if IN_SAGEMAKER else "pipelines/lead_conversion_rate/model/pickles/data_processed.pkl"
    
    if not os.path.exists (processed_data_path):
        logger.error(f'File not found: {processed_data_path}')
        sys.exit(1)

    logger.info(f'Cargando datos procesades desde {processed_data_path}')
    
    transformed_data = pd.read_pickle(processed_data_path)        
    
    

    # Initialize model and fit
    model = Model()
    best_models, features = model.fit(transformed_data)

    # Save the best models and features
    save_models(best_models, model.name)
    save_features(transformed_data, features, model.name)


if __name__ == "__main__":
    fit()
    