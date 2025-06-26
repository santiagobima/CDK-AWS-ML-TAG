import json
import os
import joblib
import pandas as pd
from pipelines.lead_conversion_rate.model.utls.utls import config, save_config
import sys
import subprocess

# Instalación del paquete durante ejecución en SageMaker
if os.path.exists("/opt/ml/processing/source_code"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "/opt/ml/processing/source_code"])
    sys.path.insert(0, "/opt/ml/processing/source_code")




def get_stage_features(stage, summary_file=None, exclude_type=None, get_categorical=False):
    """
    Extracts features from the summary DataFrame based on the specified stage and type.

    Parameters:
    summary_file (str, optional): The file path of the summary CSV containing feature information.
    stage (str): The stage name ('init_stage', 'mid_stage', or 'final_stage') to filter
                    the features.
    exclude_type (str, optional): The data type to exclude from the features.
                    Defaults to "datetime64[ns]".

    Returns:
    list: A list of feature names that match the criteria.
    """
    summary_file = summary_file or config['Utls']['get_stage_features']['summary_file']
    exclude_type = exclude_type or config['Utls']['get_stage_features']['exclude_type']

    if summary_file is None or stage is None:
        raise ValueError(
            "Both 'summary_file' and 'stage' must be provided,"
            "either as parameters or in the config.")

    summary_baseline = pd.read_csv(summary_file, index_col=False, delimiter=',')

    if stage == 'init_stage':
        features = (summary_baseline[
            (summary_baseline.stage > 0) & (summary_baseline.stage <= 1) & (
                summary_baseline.Type != exclude_type) & (summary_baseline['In use'])]
            .Column.tolist())
    elif stage == 'mid_stage':
        features = (summary_baseline[
            (summary_baseline.stage > 0) & (summary_baseline.stage <= 2) & (
                summary_baseline.Type != exclude_type) & (summary_baseline['In use'])]
            .Column.tolist())
    elif stage == 'final_stage':
        features = [
            feature for feature in summary_baseline[
                (summary_baseline.stage > 0) & (summary_baseline.stage <= 3) & (
                    summary_baseline.Type != exclude_type) & (summary_baseline['In use'])]
            .Column.tolist()
            if not feature.startswith('feat_init')
        ]

    else:
        raise ValueError(
            "Invalid stage. Please choose from 'init_stage', 'mid_stage', or 'final_stage'.")

    if get_categorical:
        categorical_features = summary_baseline[
            summary_baseline['Feat_type'] == 'categorical'].index.tolist()
        return features, categorical_features

    return features


def get_categorical_features(summary_file=None):
    """
    Retrieve categorical feature indices from a summary file.

    Parameters:
        summary_file (str): Path to the summary file.
        If None, uses config['Utls']['get_stage_features']['summary_file'].

    Returns:
        list: List of indices corresponding to categorical features.
    """
    summary_file = summary_file or config['Utls']['get_stage_features']['summary_file']
    if summary_file is None:
        raise ValueError("'summary_file' must be provided, either as parameters or in the config.")

    summary_baseline = pd.read_csv(summary_file, index_col=False, delimiter=',')
    categorical_features = summary_baseline[
        summary_baseline['Feat_type'] == 'categorical'].index.tolist()
    return categorical_features


def load_model(name, stage):
    """
    Load a trained model from disk.

    Parameters:
        name (str): Name of the model.
        stage (str): Stage of the model.

    Returns:
        obj: Loaded model object.
    """
    model_path = os.path.join(config['Model']['save_model_path'], stage, name + ".joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return joblib.load(model_path)


def save_models(models, name):
    """
    Save multiple trained models to disk.

    Parameters:
        models (dict): Dictionary containing models with keys as stages.
        name (str): Name prefix for model files.
    """
    for key, value in models.items():
        model_dir = os.path.join(config['Model']['save_model_path'], key)
        os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
        model_path = os.path.join(model_dir, name + ".joblib")
        joblib.dump(value, model_path)

        config_dir = os.path.join(model_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)  # Create directory if it doesn't exist
        config_path = os.path.join(config_dir, "model_config.yml")

        save_config(config, config_path)


def save_features(data, features, name):
    """
    Save feature information to JSON files.

    Parameters:
        data: the data tha was used for training, we need this info to save the dtypes of features
        features (dict): Dictionary containing features with keys as stages.
        name (str): Name prefix for feature files.
    """
    for key, value in features.items():
        feat = data[value].dtypes.apply(lambda x: x.name).to_dict()
        features_dir = os.path.join(config['Model']['save_features_path'], key)
        os.makedirs(features_dir, exist_ok=True)  # Create directory if it doesn't exist
        features_path = os.path.join(features_dir, name + ".json")
        with open(features_path, 'w') as f:
            json.dump(feat, f)


def load_features(name, stage):
    """
    Load feature information from JSON files.

    Parameters:
        name (str): Name of the feature file.
        stage (str): Stage of the features.

    Returns:
        dict: Loaded feature information.
    """
    features_path = os.path.join(config['Model']['save_features_path'], stage, name + ".json")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file '{features_path}' not found.")
    with open(features_path, 'r') as file:
        data = json.load(file)
    return data


def write_prediction(probabilities, name, stage, locally=True):
    """
    Write model predictions to a CSV file.

    Parameters:
        probabilities (array-like): Predicted probabilities.
        name (str): Name of the model.
        stage (str): Stage of the predictions.
        locally (bool): If True, save results locally.

    Returns:
        pd.DataFrame: DataFrame containing the saved predictions.
    """
    if locally:
        results_df = pd.DataFrame(probabilities, columns=["probabilities"])
        filename_path = os.path.join(config['Model'].get('save_model_results'), stage)
        os.makedirs(filename_path, exist_ok=True)
        filename = os.path.join(filename_path, name + ".csv")

        results_df.to_csv(filename, index=False)

        return results_df
